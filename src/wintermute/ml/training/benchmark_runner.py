from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wintermute.ml.evaluation.lm_eval.config import LmEvalConfig
from wintermute.ml.evaluation.lm_eval.harness import LmEvalHarness
from wintermute.ml.evaluation.lm_eval.runner import LmEvalExecution, LmEvalResult
from wintermute.ml.evaluation.lm_eval.targets import (
    build_loaded_model_adapter,
    build_live_run_target,
)
from wintermute.ml.models.base import BaseModel
from wintermute.ml.runs import RunCheckpoints
from wintermute.ml.runs.checkpoints import CheckPoint
from wintermute.ml.runs.store import RUN_STORE_DIRNAME
from wintermute.ml.tasks.baseline import BaselineSuite
from wintermute.ml.tasks.factory import TaskWrapper
from wintermute.ml.tasks.implementations.causal_lm.task import CausalLmTaskWrapper
from wintermute.ml.training.config import BenchmarkConfig
from wintermute.ml.training.logger import Logger
from wintermute.ml.training.metrics import TrainingMetricsLogger
from wintermute.tools.logging import console_log
from wintermute.tools.model import preserving_rng_state, unload_from_gpu


class TrainingBenchmarkRunner:
    """Run configured lm-eval benchmarks against periodic training checkpoints."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        model: BaseModel,
        task_wrapper: TaskWrapper,
        checkpoints: RunCheckpoints,
        config: BenchmarkConfig,
        metrics_logger: TrainingMetricsLogger,
        logger: Logger,
    ) -> None:
        if not isinstance(task_wrapper, CausalLmTaskWrapper):
            raise ValueError(
                "benchmark_config is supported only for causal language-model training"
            )
        output_root_path = Path(output_root)
        self.run_id = checkpoints.run_id
        self.run_path = output_root_path / RUN_STORE_DIRNAME / self.run_id
        self.model = model
        self.task_wrapper = task_wrapper
        self.checkpoints = checkpoints
        self.config = config
        self.metrics_logger = metrics_logger
        self.logger = logger
        self.execution = LmEvalExecution(
            output_root=output_root_path,
            config=LmEvalConfig.from_mapping(
                config.values,
                default_device=task_wrapper.device,
            ),
        )
        self._checkpoint_count = 0

    def reconfigure_cadence(self, every_n_checkpoints: int) -> bool:
        normalized = max(1, every_n_checkpoints)
        if normalized == self.config.every_n_checkpoints:
            return False
        self.config.every_n_checkpoints = normalized
        self._checkpoint_count = 0
        return True

    def run_baselines(
        self,
        baseline_suite: BaselineSuite,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        results: dict[str, dict[str, dict[str, Any]]] = {}
        harness = LmEvalHarness(self.execution.output_root)
        for evaluator in baseline_suite.benchmark_evaluators():
            if not isinstance(evaluator.task_wrapper, CausalLmTaskWrapper):
                raise ValueError(
                    "lm-eval baselines require causal language-model task wrappers"
                )

            console_log(
                "benchmark",
                f"evaluating baseline '{evaluator.name}' with lm-eval",
            )
            try:
                with preserving_rng_state():
                    adapter = build_loaded_model_adapter(
                        model_name=evaluator.name,
                        model=evaluator.model,
                        task_wrapper=evaluator.task_wrapper,
                    )
                    raw_results = harness.evaluate(
                        adapter,
                        self.execution.config,
                    ).results
                    self._merge_baseline_results(
                        results,
                        baseline_name=evaluator.name,
                        raw_results=raw_results,
                    )
            finally:
                evaluator.model = unload_from_gpu(evaluator.model)
        return results

    @staticmethod
    def _merge_baseline_results(
        destination: dict[str, dict[str, dict[str, Any]]],
        *,
        baseline_name: str,
        raw_results: Mapping[str, Any],
    ) -> None:
        for collection_name in ("results", "groups"):
            collection = raw_results.get(collection_name)
            if not isinstance(collection, Mapping):
                continue
            for task_name, raw_metrics in collection.items():
                if not isinstance(raw_metrics, Mapping):
                    continue
                destination.setdefault(str(task_name), {})[baseline_name] = dict(
                    raw_metrics
                )

    def run_after_checkpoint(self, checkpoint: CheckPoint) -> LmEvalResult | None:
        self._checkpoint_count += 1
        if self._checkpoint_count % self.config.every_n_checkpoints != 0:
            return None

        resolved_checkpoint = self.checkpoints.resolve(
            checkpoint.checkpoint_id,
            with_states=False,
        )
        if resolved_checkpoint is None:
            raise RuntimeError(
                f"saved checkpoint '{checkpoint.checkpoint_id}' could not be "
                "resolved for benchmarking"
            )

        self.logger.log_text(
            "benchmark",
            f"starting lm-eval benchmarks for checkpoint '{checkpoint.checkpoint_id}'",
            step=checkpoint.global_step,
        )
        started_at = time.perf_counter()
        was_training = self.model.training
        try:
            with preserving_rng_state():
                target = build_live_run_target(
                    run_id=self.run_id,
                    run_path=self.run_path,
                    model=self.model,
                    task_wrapper=self.task_wrapper,
                    resolved_checkpoint=resolved_checkpoint,
                )
                result = self.execution.run_resolved(target)
        finally:
            if was_training:
                self.model.train()
            else:
                self.model.eval()

        elapsed_sec = time.perf_counter() - started_at
        self.metrics_logger.log_benchmark(
            results=result.results,
            global_step=checkpoint.global_step,
        )
        self.logger.log_scalar(
            "perf/benchmark_checkpoint_sec",
            elapsed_sec,
            checkpoint.global_step,
        )
        self.logger.log_text(
            "benchmark",
            f"completed lm-eval benchmarks for checkpoint '{checkpoint.checkpoint_id}' "
            f"in {elapsed_sec:.1f}s; results: {result.results_path.as_posix()}",
            step=checkpoint.global_step,
        )
        return result
