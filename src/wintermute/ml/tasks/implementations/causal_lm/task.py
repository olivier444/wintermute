from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, cast

import torch
import torch.nn as nn

from wintermute.data.iterate.dataclasses import TrainingExample, TrainingViewConfig
from wintermute.data.iterate.training_view import (
    AbstractTrainingView,
    PackedTrainingView,
    build_base_training_view,
)
from wintermute.ml.runs import Run
from wintermute.ml.tasks.baseline import BaselineSuite
from wintermute.ml.inference.conversation import (
    InferenceSession,
    InferenceTurn,
)
from wintermute.ml.tasks.factory import TaskWrapper
from wintermute.ml.tasks.implementations.causal_lm.baseline import CausalLmBaselineSuite
from wintermute.ml.tasks.implementations.causal_lm.config import CausalLmTaskConfig
from wintermute.ml.tasks.implementations.causal_lm.generation import CausalLmTokenGenerator
from wintermute.ml.tasks.implementations.causal_lm.objective import CausalLmNtpObjective
from wintermute.ml.tasks.implementations.causal_lm.probes.executor import (
    CausalLmProbeExecutor,
)
from wintermute.ml.tasks.implementations.causal_lm.probes.monitor import (
    CausalLmProbeMonitor,
)
from wintermute.ml.tasks.implementations.causal_lm.backend import CausalLmBackend
from wintermute.ml.tokenization.chat_format import ChatFormat
from wintermute.ml.tokenization.wrapper import TokenizerWrapper
from wintermute.ml.training.logger import Logger
from wintermute.ml.training.monitor import TrainingMonitor
from wintermute.ml.training.view import (
    CompositeSourceExhaustionObserver,
    TrainingViewIterationReport,
)


class CausalLmTaskWrapper(TaskWrapper):
    """Configured causal-LM adapter for supervised NTP training."""

    def __init__(
            self,
            *,
            device: torch.device,
            bf_16_enabled: bool,
            params: Dict[str, Any],
            root_path: str,
        ):
        super().__init__(device)
        cfg = CausalLmTaskConfig.from_dict(params)

        self.params = dict(params)
        self.root_path = root_path
        self.backend = CausalLmBackend.from_config(
            config=cfg,
            root_path=root_path,
            device=device,
            bf16_enabled=bf_16_enabled,
        )
        self.packing_policy = cfg.packing_policy
        self.windowing_policy = cfg.windowing_policy
        self.use_task_labels = cfg.use_task_labels
        self.tag_loss_weight = cfg.tag_loss_weight
        self.verbosity_labels = cfg.verbosity_labels
        self.format_labels = cfg.format_labels
        self.bf_16_enabled = bf_16_enabled
        self.objective = CausalLmNtpObjective(
            device=device,
            bf16_enabled=bf_16_enabled,
            tokenizer=self.backend.tokenizer,
            eval_position_boundaries=cfg.eval_position_boundaries,
            completion_position_weighting=cfg.completion_position_weighting,
            completion_prefix_masking=cfg.completion_prefix_masking,
        )

    @property
    def tokenizer(self) -> TokenizerWrapper:
        return self.backend.tokenizer

    @property
    def chat(self) -> ChatFormat:
        return self.backend.chat_format

    @property
    def generator(self) -> CausalLmTokenGenerator:
        return self.backend.generator

    def start_inference_session(
        self,
        model: nn.Module,
        system_prompt: str | None = None,
        initial_turns: tuple[InferenceTurn, ...] = (),
    ) -> InferenceSession:
        return self.backend.start_inference_session(
            model,
            system_prompt=system_prompt,
            initial_turns=initial_turns,
        )


    def build_training_view(
            self,
            output_root: str,
            training_view_config: TrainingViewConfig,
            silent: bool = False,
            source_exhaustion_observer: CompositeSourceExhaustionObserver | None = None,
        ) -> Iterable[TrainingExample]:
        view = build_base_training_view(
            tokenizer=self.backend.tokenizer,
            chat_format=self.backend.chat_format,
            training_view_config=training_view_config,
            windowing_policy=self.windowing_policy,
            output_root=output_root,
            use_task_labels=self.use_task_labels,
            tag_loss_weight=self.tag_loss_weight,
            verbosity_labels=self.verbosity_labels,
            format_labels=self.format_labels,
            source_exhaustion_observer=source_exhaustion_observer,
            silent=silent,
        )

        if self.packing_policy.enabled:
            view = PackedTrainingView(view, self.packing_policy, silent)

        return self.objective.wrap_training_view(view)

    def training_view_iteration_report(
        self,
        view: Iterable[TrainingExample],
    ) -> TrainingViewIterationReport | None:
        if not isinstance(view, AbstractTrainingView):
            return None

        stats = view.last_iteration_stats
        if not stats.has_sft_activity():
            return None

        return TrainingViewIterationReport(
            summary=f"sft stats: {stats.describe()}",
            metrics=stats.metrics(),
        )


    def build_baseline_suite(self, baseline_config: Mapping[str, Any]) -> BaselineSuite:
        return CausalLmBaselineSuite(
            wrapper_builder=self._build_hf_baseline_wrapper,
            run_wrapper_builder=self._build_run_baseline_wrapper,
            baseline_config=baseline_config,
            output_root=Path(self.root_path),
        )


    def build_training_monitor(
        self,
        *,
        model: nn.Module,
        logger: Logger,
        configs: Iterable[Mapping[str, Any]],
    ) -> TrainingMonitor:
        return CausalLmProbeMonitor.from_configs(
            executor=CausalLmProbeExecutor(
                tokenizer=self.backend.tokenizer,
                chat_format=self.backend.chat_format,
                generator=self.backend.generator,
            ),
            model=model,
            logger=logger,
            configs=configs,
        )


    def export_assets(self, output_dir: str | Path) -> None:
        self.backend.export_assets(output_dir)


    def _build_hf_baseline_wrapper(self, model_name: str) -> TaskWrapper:
        task_params = dict(self.params)
        task_params.pop("tokenizer_id", None)
        task_params.pop("tokenizer_file", None)
        task_params.pop("completion_prefix_masking", None)
        task_params["hf_tokenizer_name"] = model_name

        wrapper_cls = cast(Any, type(self))
        return wrapper_cls(
            device=self.device,
            bf_16_enabled=self.bf_16_enabled,
            params=task_params,
            root_path=self.root_path,
        )


    def _build_run_baseline_wrapper(self, run: Run) -> TaskWrapper:
        task_params = dict(run.config.task_config.params)
        tokenizer_file = run.model_path / "tokenizer.json"
        if tokenizer_file.is_file():
            task_params.pop("tokenizer_id", None)
            task_params.pop("hf_tokenizer_name", None)
            task_params["tokenizer_file"] = tokenizer_file.as_posix()

        wrapper_cls = cast(Any, type(self))
        return wrapper_cls(
            device=self.device,
            bf_16_enabled=self.bf_16_enabled,
            params=task_params,
            root_path=self.root_path,
        )


    def compute_loss(
        self,
        model: nn.Module,
        examples: List[TrainingExample],
    ) -> torch.Tensor:
        return self.objective.compute_loss(model, examples)

    def compute_eval_metrics(
        self,
        model: nn.Module,
        batches: Iterable[List[TrainingExample]],
    ) -> Dict[str, float]:
        return self.objective.compute_eval_metrics(model, batches)
