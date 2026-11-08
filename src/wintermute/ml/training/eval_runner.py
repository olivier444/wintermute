from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn

from wintermute.ml.evaluation.limits import limit_eval_examples
from wintermute.ml.training.eval_suite import EvalStepResult, RuntimeEvalSuite
from wintermute.ml.training.metrics import TrainingMetricsLogger
from wintermute.tools.model import evaluating, iter_batches


class EvaluationRunner:
    def __init__(
        self,
        *,
        output_root: str,
        model: nn.Module,
        metrics_logger: TrainingMetricsLogger,
        suites: Iterable[RuntimeEvalSuite],
    ) -> None:
        self.output_root = output_root
        self.model = model
        self.metrics_logger = metrics_logger
        self.suites = list(suites)
        self.primary_suite = next(suite for suite in self.suites if suite.is_primary)

    def select_due_suites(self) -> list[RuntimeEvalSuite]:
        return [suite for suite in self.suites if suite.is_evaluation_due()]

    def run_due(self, *, global_step: int) -> EvalStepResult:
        return self.run(
            self.select_due_suites(),
            global_step=global_step,
        )

    def run_all(self, *, global_step: int) -> EvalStepResult:
        return self.run(self.suites, global_step=global_step)

    def run(
        self,
        suites: Iterable[RuntimeEvalSuite],
        *,
        global_step: int,
    ) -> EvalStepResult:
        metrics_by_suite: dict[str, dict[str, float]] = {}

        for suite in suites:
            metrics = self._compute_suite(suite)
            metrics_by_suite[suite.name] = metrics
            self.metrics_logger.log_eval(
                suite_name=suite.name,
                metrics=metrics,
                global_step=global_step,
            )

        return EvalStepResult(
            metrics_by_suite=metrics_by_suite,
            primary_suite_name=self.primary_suite.name,
        )

    def _compute_suite(self, suite: RuntimeEvalSuite) -> dict[str, float]:
        eval_view = suite.task_wrapper.build_training_view(
            output_root=self.output_root,
            training_view_config=suite.view_config,
            silent=True,
        )

        with evaluating(self.model):
            metrics = suite.task_wrapper.compute_eval_metrics(
                self.model,
                iter_batches(
                    limit_eval_examples(eval_view, suite.max_eval_units),
                    suite.batch_size,
                ),
            )

        if "loss" not in metrics:
            raise RuntimeError(f"Eval suite '{suite.name}' did not produce a 'loss' metric.")
        return metrics
