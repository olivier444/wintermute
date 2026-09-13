from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

from wintermute.data.iterate.training_view import TrainingViewIterationStats
from wintermute.ml.training.logger import Logger
from wintermute.ml.training.optimizer_diagnostics import (
    FirstUpdateDiagnostics,
    OptimizerDiagnosticsResult,
    UpdatePairDiagnostics,
)
from wintermute.ml.training.runtime import MacroStepContext
from wintermute.ml.training.scaling_law import ScalingLawFit
from wintermute.tools.model import log_cuda_memory


class TrainingMetricsLogger:
    def __init__(self, logger: Logger, *, trainable_param_count: int) -> None:
        self.logger = logger
        self.trainable_param_count = trainable_param_count

    def log_update(
        self,
        *,
        global_step: int,
        epoch: int,
        units_seen: int,
        examples_seen: int,
        last_train_loss: float,
        total_units: int,
        learning_rate: float,
        grad_accum_size_units: int,
        diagnostics: OptimizerDiagnosticsResult,
    ) -> None:
        self.logger.log_scalar("progress/train_epoch", epoch, global_step)
        self.logger.log_scalar("loss/train_last_batch", last_train_loss, global_step)
        self.logger.log_scalar("progress/units_seen", float(units_seen), global_step)
        self.logger.log_scalar(
            "progress/examples_seen", float(examples_seen), global_step
        )
        self.logger.log_scalar(
            "progress/pct",
            min(1.0, float(units_seen / total_units)),
            global_step,
        )
        self.logger.log_scalar("optimizer/learning_rate", learning_rate, global_step)
        self.logger.log_scalar(
            "optimizer/grad_accum_size_units",
            float(grad_accum_size_units),
            global_step,
        )
        grad_norm = diagnostics.last_grad_norm
        grad_rms = grad_norm / math.sqrt(self.trainable_param_count)
        self.logger.log_scalar("optimizer/grad_norm", grad_norm, global_step)
        self.logger.log_scalar("optimizer/grad_rms", grad_rms, global_step)
        self.logger.log_scalar(
            "optimizer/adam_sqrt_v_rms", diagnostics.adam_sqrt_v_rms, global_step
        )
        self.logger.log_scalar(
            "optimizer/adam_m_over_sqrt_v_rms",
            diagnostics.adam_m_over_sqrt_v_rms,
            global_step,
        )

    def log_macro_step(
        self,
        *,
        global_step: int,
        units_seen: int,
        context: MacroStepContext,
        diagnostics: OptimizerDiagnosticsResult,
    ) -> None:
        self.logger.log_scalar(
            "loss/train", context.compute_macro_train_loss(), global_step
        )
        if diagnostics.first_update is None:
            raise RuntimeError("Macro step is missing first-update diagnostics.")
        self._log_first_update(diagnostics.first_update, global_step=global_step)
        if diagnostics.update_pair is not None:
            self._log_update_pair(diagnostics.update_pair, global_step=global_step)

        self.logger.log_scalar(
            "optimizer/grad_clip_fraction", diagnostics.grad_clip_fraction, global_step
        )
        self.logger.log_scalar(
            "grad_noise/update_batch_count_available",
            float(diagnostics.gradient_noise_available_update_batches),
            global_step,
        )
        self.logger.log_scalar(
            "grad_noise/update_batch_count_required",
            float(diagnostics.gradient_noise_required_update_batches),
            global_step,
        )
        self.logger.log_scalar(
            "grad_noise/modulus", float(diagnostics.gradient_noise_modulus), global_step
        )
        self.logger.log_scalar(
            "grad_noise/estimation_due",
            float(diagnostics.gradient_noise_estimation_due),
            global_step,
        )
        if diagnostics.gradient_noise_estimate is not None:
            estimate = diagnostics.gradient_noise_estimate
            self.logger.log_scalar(
                "grad_noise/noise_ratio", estimate.noise_ratio(), global_step
            )
            self.logger.log_scalar(
                "grad_noise/mean_sq_distance",
                estimate.mean_squared_distance,
                global_step,
            )
            self.logger.log_scalar(
                "grad_noise/mean_grad_norm", estimate.mean_grad_norm, global_step
            )
            self.logger.log_scalar(
                "grad_noise/mean_update_batch_grad_norm",
                estimate.mean_update_batch_grad_norm,
                global_step,
            )
            self.logger.log_scalar(
                "grad_noise/avg_update_batch_units",
                estimate.avg_update_batch_units,
                global_step,
            )
            self.logger.log_scalar(
                "grad_noise/critical_batch_size_units",
                estimate.critical_batch_size_units(),
                global_step,
            )
            self.logger.log_scalar(
                "perf/gradient_noise_sec", estimate.elapsed_sec, global_step
            )
        self.logger.log_scalar(
            "perf/eval_checkpoint_sec", context.eval_checkpoint_sec, global_step
        )
        self.logger.log_scalar(
            "perf/monitor_checkpoint_sec", context.monitor_checkpoint_sec, global_step
        )

        log_cuda_memory(self.logger, global_step)

        units_per_sec, sec_per_step = context.compute_perf_metrics(
            units_seen=units_seen,
            global_step=global_step,
        )
        self.logger.log_scalar("perf/units_per_sec", units_per_sec, global_step)
        self.logger.log_scalar("perf/sec_per_step", sec_per_step, global_step)

    def log_eval(
        self,
        *,
        suite_name: str,
        metrics: Mapping[str, float],
        global_step: int,
    ) -> None:
        for metric, value in metrics.items():
            self.logger.log_scalar(
                f"{metric}/eval/{suite_name}", value, global_step
            )

    def log_scaling_law(
        self,
        *,
        fit: ScalingLawFit,
        global_step: int,
    ) -> None:
        for name, value in fit.metrics().items():
            self.logger.log_scalar(name, value, global_step)

    def log_benchmark(
        self,
        *,
        results: Mapping[str, object],
        global_step: int,
    ) -> None:
        for collection_name in ("results", "groups"):
            task_results = results.get(collection_name)
            if not isinstance(task_results, Mapping):
                continue

            for task_name, raw_metrics in task_results.items():
                if not isinstance(raw_metrics, Mapping):
                    continue
                for raw_metric_name, raw_value in raw_metrics.items():
                    if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                        continue
                    metric_name, separator, filter_name = str(
                        raw_metric_name
                    ).partition(",")
                    filter_suffix = (
                        f"/{filter_name}"
                        if separator and filter_name and filter_name != "none"
                        else ""
                    )
                    self.logger.log_scalar(
                        f"{metric_name}/benchmark/{task_name}{filter_suffix}",
                        float(raw_value),
                        global_step,
                    )

    def log_training_view(
        self,
        stats: TrainingViewIterationStats,
        *,
        global_step: int,
    ) -> None:
        total_candidates = max(1, stats.sft_candidates)
        scratchpad_examples = max(1, stats.sft_examples_with_scratchpad)
        system_prompt_examples = max(1, stats.sft_examples_with_system_prompt)
        history_examples = max(1, stats.sft_examples_with_history)
        values = {
            "data/sft/emitted_pct": 100.0 * stats.sft_emitted / total_candidates,
            "data/sft/rejected_pct": 100.0 * stats.sft_rejected / total_candidates,
            "data/sft/scratchpad_removed_pct": (
                100.0 * stats.sft_scratchpad_removed / total_candidates
            ),
            "data/sft/history_reduced_pct": (
                100.0 * stats.sft_history_reduced / total_candidates
            ),
            "data/sft/history_turns_dropped_total": float(
                stats.sft_history_turns_dropped_total
            ),
            "data/sft/history_turns_dropped_avg": (
                stats.sft_history_turns_dropped_total / history_examples
            ),
            "data/sft/system_prompt_dropped_pct": (
                100.0 * stats.sft_system_prompt_dropped / total_candidates
            ),
            "data/sft/scratchpad_removed_pct_of_scratchpad_examples": (
                100.0 * stats.sft_scratchpad_removed / scratchpad_examples
            ),
            "data/sft/system_prompt_dropped_pct_of_system_prompt_examples": (
                100.0 * stats.sft_system_prompt_dropped / system_prompt_examples
            ),
            "data/sft/reject_required_turn_too_long_pct": (
                100.0 * stats.sft_reject_required_turn_too_long / total_candidates
            ),
            "data/sft/reject_invalid_example_pct": (
                100.0 * stats.sft_reject_invalid_example / total_candidates
            ),
        }
        for name, value in values.items():
            self.logger.log_scalar(name, value, global_step)

    def _log_first_update(
        self,
        diagnostics: FirstUpdateDiagnostics,
        *,
        global_step: int,
    ) -> None:
        values = {
            "optimizer/param_norm": diagnostics.param_norm,
            "optimizer/delta_param_norm": diagnostics.delta_param_norm,
            "optimizer/delta_param_to_param": diagnostics.delta_param_to_param,
            "optimizer/grad_to_param": diagnostics.grad_to_param,
            "optimizer/neg_grad_update_cos": diagnostics.neg_grad_update_cos,
            "optimizer/grad_dot_update": diagnostics.grad_dot_update,
            "optimizer/observed_loss_decrease": diagnostics.observed_loss_decrease,
            "optimizer/observed_to_predicted_loss_decrease_ratio": (
                diagnostics.observed_to_predicted_loss_decrease_ratio
            ),
        }
        for name, value in values.items():
            self.logger.log_scalar(name, value, global_step)

    def _log_update_pair(
        self,
        diagnostics: UpdatePairDiagnostics,
        *,
        global_step: int,
    ) -> None:
        self.logger.log_scalar(
            "optimizer/grad_cos_prev", diagnostics.grad_cos_prev, global_step
        )
        self.logger.log_scalar(
            "optimizer/update_cos_prev", diagnostics.update_cos_prev, global_step
        )
