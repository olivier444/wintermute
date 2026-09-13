from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from wintermute.ml.runs import Run, RunStore, SourceCapture, capture_run_source
from wintermute.ml.training.config import EvaluationConfig
from wintermute.ml.training.controller import (
    CompositeTrainerController,
    FileTrainerController,
    SlackTrainerController,
    TrainerController,
    slack_command_help_hint,
)
from wintermute.ml.training.eval_suite import RuntimeEvalSuite
from wintermute.ml.training.logger import CompositeLogger, Logger, SlackRunLogger, TensorBoardRunLogger
from wintermute.ml.training.trainer import Trainer
from wintermute.ml.tasks.factory import TaskWrapper, TaskWrapperFactory
from wintermute.ml.training.task_specification import TaskSpecification
from wintermute.tools.params import get_bool
from wintermute.tools.slack.runtime import SlackBoltRuntime


def _uses_task_labels(task_config: TaskSpecification) -> bool:
    return get_bool(task_config.params, "use_task_labels", False)


def _uses_derived_labels(task_config: TaskSpecification, label_name: str) -> bool:
    return task_config.params.get(f"{label_name}_labels") is not None


class TrainingWorkspace:
    def __init__(
        self,
        run: Run,
        runs: RunStore,
    ) -> None:
        self.run = run
        self.run_id = run.run_id
        self.root_path = run.root
        self.run_dir = run.path
        self.manifest = run.manifest
        self.checkpoints = run.checkpoints
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._slack_runtime: Optional[SlackBoltRuntime] = None
        self.logger: Optional[Logger] = None
        self.run_controller: Optional[TrainerController] = None
        self.source_capture: SourceCapture | None = None
        self._runs = runs
        self._wrapper_factory = TaskWrapperFactory()

    @property
    def config(self):
        return self.run.config

    @property
    def spec(self):
        return self.run.spec

    def create_trainer(self, *, resume_checkpoint: str = "last") -> Trainer:
        if self.logger is None or self.run_controller is None:
            self.logger, self.run_controller = self._build_runtime_adapters()
        assert self.logger is not None
        assert self.run_controller is not None

        reference_run_id = self.config.reference_run_id
        model_source = self._resolve_model_source(reference_run_id)
        self._seed_model_initialization()
        model = self.run.build_model()
        reference_optimizer_state = None
        if model_source is not None:
            reference_checkpoint = self.config.reference_checkpoint
            if reference_checkpoint is None:
                model_source.try_load_into(model)
            else:
                checkpoint = model_source.load_checkpoint_into(model, reference_checkpoint)
                if self.config.load_reference_optimizer_state:
                    reference_optimizer_state = checkpoint.states.get("optimizer")
                    if reference_optimizer_state is None:
                        raise ValueError(
                            f"checkpoint '{checkpoint.checkpoint_id}' for reference run "
                            f"'{model_source.run_id}' has no optimizer state"
                        )
        bf16_enabled = bool(self.config.trainer_config.use_bf16 and self.device.type == "cuda")

        wrapper = self._wrapper_factory.build_wrapper(
            self.config.task_config,
            self.device,
            bf16_enabled,
            self.root_path,
        )
        eval_suites = [
            self._build_eval_suite(spec, bf16_enabled, self.config.trainer_config.batch_size)
            for spec in self.config.evaluation_configs
        ]
        return Trainer(
            output_root=self.root_path.as_posix(),
            run_id=self.run_id,
            run_controller=self.run_controller,
            task_wrapper=wrapper,
            eval_suites=eval_suites,
            model=model,
            config=self.config,
            checkpoints=self.checkpoints,
            logger=self.logger,
            device=self.device,
            save_model=lambda model, dirname: self.save_model(model, wrapper, dirname),
            resume_checkpoint=resume_checkpoint,
            initial_optimizer_state=reference_optimizer_state,
        )

    def _seed_model_initialization(self) -> None:
        seed = int(self.config.trainer_config.seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def save_model(
        self,
        model: nn.Module,
        task_wrapper: TaskWrapper,
        directory_name: str = "model",
        overwrite: bool = False,
    ) -> bool:
        if self.run.has_saved_model(directory_name) and not overwrite:
            return False

        self.run.save_model(
            model=model,
            overwrite=overwrite,
            directory_name=directory_name,
            code_version=(
                self.source_capture.commit
                if self.source_capture is not None
                else None
            ),
            extra_meta=(
                {"source_capture": self.source_capture.relative_path}
                if self.source_capture is not None
                else None
            ),
        )
        task_wrapper.export_assets(self.run.model_artifact_path(directory_name))
        return True

    def capture_source(
        self,
        *,
        operation: str,
        checkpoint_selector: str | None = None,
    ) -> SourceCapture:
        self.source_capture = capture_run_source(
            self.run_dir,
            operation=operation,
            checkpoint_selector=checkpoint_selector,
        )
        return self.source_capture

    @classmethod
    def open(
        cls,
        root: Path,
        run_id: str,
    ) -> "TrainingWorkspace":
        runs = RunStore(root)
        return cls(runs.open(run_id, load_slack_config=True), runs)

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()
        if self._slack_runtime is not None:
            self._slack_runtime.close()

    def __enter__(self) -> "TrainingWorkspace":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def _build_runtime_adapters(self) -> tuple[Logger, TrainerController]:
        base_logger: Logger = TensorBoardRunLogger(
            self.run_dir,
            primary_eval_suite_name=self.config.primary_evaluation_config.name,
            eval_suite_names=[spec.name for spec in self.config.evaluation_configs],
            routing_label_eval_suite_names={
                "task": [
                    spec.name
                    for spec in self.config.evaluation_configs
                    if _uses_task_labels(spec.task_config)
                ],
                "verbosity": [
                    spec.name
                    for spec in self.config.evaluation_configs
                    if _uses_derived_labels(spec.task_config, "verbosity")
                ],
                "format": [
                    spec.name
                    for spec in self.config.evaluation_configs
                    if _uses_derived_labels(spec.task_config, "format")
                ],
            },
            benchmark_task_names=(
                self.config.benchmark_config.tasks
                if self.config.benchmark_config is not None
                else []
            ),
        )
        base_controller: TrainerController = FileTrainerController(self.run_dir)

        slack_cfg = self.config.slack_config
        if slack_cfg is None:
            return base_logger, base_controller

        runtime = SlackBoltRuntime(
            bot_token=slack_cfg.bot_token,
            app_token=slack_cfg.app_token,
            run_root=self.run_dir,
            channel=slack_cfg.channel,
            root_text=(
                f"Run `{self.run_id}` attached.\n"
                f"{slack_command_help_hint()}"
            ),
        ).start()
        self._slack_runtime = runtime

        logger = CompositeLogger(
            [
                base_logger,
                SlackRunLogger(
                    runtime.session,
                    run_id=self.run_id,
                    checkpoints=self.checkpoints,
                    notify_interval_sec=slack_cfg.push_interval_sec,
                    metrics_history_macro_steps=slack_cfg.metrics_history_macro_steps,
                ),
            ]
        )
        controller = CompositeTrainerController(
            [
                base_controller,
                SlackTrainerController(runtime.commands),
            ]
        )
        return logger, controller

    def _resolve_model_source(
        self,
        reference_run_id: str | None,
    ) -> Run | None:
        if not reference_run_id:
            return None
        if reference_run_id == self.run_id:
            return self.run

        source_run = self._runs.try_open(reference_run_id)
        if source_run is None:
            raise ValueError(f"reference run '{reference_run_id}' could not be opened")
        return source_run

    def _build_eval_suite(
        self,
        spec: EvaluationConfig,
        bf16_enabled: bool,
        default_batch_size: int,
    ) -> RuntimeEvalSuite:
        wrapper = self._wrapper_factory.build_wrapper(
            spec.task_config,
            self.device,
            bf16_enabled,
            self.root_path,
        )

        return RuntimeEvalSuite(
            name=spec.name,
            task_wrapper=wrapper,
            view_config=spec.view_config,
            batch_size=spec.batch_size or default_batch_size,
            max_eval_units=spec.max_eval_units,
            is_primary=spec.is_primary,
            evaluation_modulus=spec.evaluation_modulus,
        )
