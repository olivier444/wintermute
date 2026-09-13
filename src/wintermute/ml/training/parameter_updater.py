from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch.optim import Optimizer

from wintermute.ml.training.config import TrainingConfig
from wintermute.ml.training.eval_runner import EvaluationRunner
from wintermute.ml.training.lr_schedule import LrSchedule
from wintermute.ml.training.optimizer_diagnostics import OptimizerDiagnosticsTracker
from wintermute.tools.params import (
    get_int,
    get_optional_float,
    get_optional_int,
    get_optional_mapping,
)

if TYPE_CHECKING:
    from wintermute.ml.training.benchmark_runner import TrainingBenchmarkRunner


@dataclass(frozen=True)
class _TrainingParameterPatch:
    start_lr: float | None
    end_lr: float | None
    max_lr: float | None
    grad_clip: float | None
    max_training_units: int | None
    checkpoint_every_batch: int | None
    grad_accum_size_units: int | None
    gradient_noise_update_batch_count: int | None
    gradient_noise_modulus: int | None
    evaluation_moduli: Mapping[str, Any] | None
    benchmark_every_n_checkpoints: int | None
    warmup_length_ratio: float | None
    cos_decay_start_ratio: float | None
    beta1: float | None
    beta2: float | None
    weight_decay: float | None

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> _TrainingParameterPatch:
        return cls(
            start_lr=get_optional_float(params, "start_lr"),
            end_lr=get_optional_float(params, "end_lr"),
            max_lr=get_optional_float(params, "max_lr"),
            grad_clip=get_optional_float(params, "grad_clip"),
            max_training_units=get_optional_int(params, "max_training_units"),
            checkpoint_every_batch=get_optional_int(
                params, "checkpoint_every_batch"
            ),
            grad_accum_size_units=get_optional_int(
                params, "grad_accum_size_units"
            ),
            gradient_noise_update_batch_count=get_optional_int(
                params, "gradient_noise_update_batch_count"
            ),
            gradient_noise_modulus=get_optional_int(
                params, "gradient_noise_modulus"
            ),
            evaluation_moduli=get_optional_mapping(params, "evaluation_moduli"),
            benchmark_every_n_checkpoints=get_optional_int(
                params, "benchmark_every_n_checkpoints"
            ),
            warmup_length_ratio=get_optional_float(params, "warmup_length_ratio"),
            cos_decay_start_ratio=get_optional_float(
                params, "cos_decay_start_ratio"
            ),
            beta1=get_optional_float(params, "beta1"),
            beta2=get_optional_float(params, "beta2"),
            weight_decay=get_optional_float(params, "weight_decay"),
        )


class TrainingParameterUpdater:
    def __init__(
        self,
        *,
        config: TrainingConfig,
        optimizer: Optimizer,
        lr_schedule: LrSchedule,
        optimizer_diagnostics: OptimizerDiagnosticsTracker,
        evaluation_runner: EvaluationRunner,
        benchmark_runner: TrainingBenchmarkRunner | None = None,
    ) -> None:
        self.config = config
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.optimizer_diagnostics = optimizer_diagnostics
        self.evaluation_runner = evaluation_runner
        self.benchmark_runner = benchmark_runner

    def apply(self, params: Mapping[str, Any]) -> tuple[str, ...]:
        patch = _TrainingParameterPatch.from_mapping(params)
        normalized_evaluation_moduli = self._normalize_evaluation_moduli(
            patch.evaluation_moduli
        )
        if (
            patch.benchmark_every_n_checkpoints is not None
            and (self.config.benchmark_config is None or self.benchmark_runner is None)
        ):
            raise ValueError("Cannot update benchmark cadence: benchmarks are disabled.")
        messages: list[str] = []

        self._apply_runtime_config(patch)
        messages.extend(self._apply_gradient_noise_config(patch))
        messages.extend(
            self._apply_evaluation_moduli(normalized_evaluation_moduli)
        )
        messages.extend(
            self._apply_benchmark_cadence(patch.benchmark_every_n_checkpoints)
        )
        self._apply_optimizer_config_patch(patch)
        self._apply_lr_config(patch)
        return tuple(messages)

    def _apply_benchmark_cadence(
        self,
        every_n_checkpoints: int | None,
    ) -> list[str]:
        if every_n_checkpoints is None:
            return []
        assert self.benchmark_runner is not None

        if not self.benchmark_runner.reconfigure_cadence(every_n_checkpoints):
            return []
        return [
            "benchmark cadence reconfigured: "
            f"every_n_checkpoints={self.benchmark_runner.config.every_n_checkpoints} "
            "(schedule reset)"
        ]

    def apply_optimizer_config(self) -> None:
        optimizer_config = self.config.optimizer_config
        betas = (optimizer_config.beta1, optimizer_config.beta2)
        for param_group in self.optimizer.param_groups:
            param_group["betas"] = betas
            param_group["eps"] = optimizer_config.eps
            param_group["weight_decay"] = (
                optimizer_config.weight_decay
                if param_group.get("apply_weight_decay", True)
                else 0.0
            )

    def _apply_runtime_config(self, patch: _TrainingParameterPatch) -> None:
        optimizer_config = self.config.optimizer_config
        trainer_config = self.config.trainer_config

        if patch.grad_clip is not None:
            optimizer_config.grad_clip = patch.grad_clip
        if patch.max_training_units is not None:
            trainer_config.max_training_units = max(1, patch.max_training_units)
        if patch.checkpoint_every_batch is not None:
            trainer_config.checkpoint_every_batch = max(
                1, patch.checkpoint_every_batch
            )
        if patch.grad_accum_size_units is not None:
            trainer_config.grad_accum_size_units = max(
                1, patch.grad_accum_size_units
            )

    def _apply_gradient_noise_config(
        self,
        patch: _TrainingParameterPatch,
    ) -> list[str]:
        messages: list[str] = []
        trainer_config = self.config.trainer_config

        if patch.gradient_noise_update_batch_count is not None:
            previous_count = trainer_config.gradient_noise_update_batch_count
            self.optimizer_diagnostics.reconfigure_gradient_noise_update_batch_count(
                patch.gradient_noise_update_batch_count
            )
            normalized_count = (
                self.optimizer_diagnostics.gradient_noise_update_batch_count
            )
            trainer_config.gradient_noise_update_batch_count = normalized_count
            if previous_count != normalized_count:
                messages.append(
                    "gradient noise history reconfigured: "
                    f"update_batch_count={normalized_count} (history reset)"
                )

        if (
            patch.gradient_noise_modulus is not None
            and self.optimizer_diagnostics.reconfigure_gradient_noise_modulus(
                patch.gradient_noise_modulus
            )
        ):
            normalized_modulus = self.optimizer_diagnostics.gradient_noise_modulus
            trainer_config.gradient_noise_modulus = normalized_modulus
            messages.append(
                "gradient noise cadence reconfigured: "
                f"modulus={normalized_modulus} (schedule reset)"
            )

        return messages

    def _normalize_evaluation_moduli(
        self,
        moduli: Mapping[str, Any] | None,
    ) -> dict[str, int]:
        if moduli is None:
            return {}

        suites_by_name = {
            suite.name: suite for suite in self.evaluation_runner.suites
        }
        unknown_names = sorted(set(moduli) - set(suites_by_name))
        if unknown_names:
            raise ValueError(f"Unknown evaluation suite(s): {', '.join(unknown_names)}.")

        normalized = {
            name: max(1, get_int(moduli, name, 1))
            for name in moduli
        }
        for name, modulus in normalized.items():
            if suites_by_name[name].is_primary and modulus != 1:
                raise ValueError(
                    "The primary evaluation suite must use evaluation_modulus=1."
                )
        return normalized

    def _apply_evaluation_moduli(self, moduli: Mapping[str, int]) -> list[str]:
        if not moduli:
            return []

        suites_by_name = {
            suite.name: suite for suite in self.evaluation_runner.suites
        }
        configs_by_name = {
            config.name: config for config in self.config.evaluation_configs
        }
        changed: list[str] = []
        for name, modulus in moduli.items():
            if not suites_by_name[name].reconfigure_modulus(modulus):
                continue
            configs_by_name[name].evaluation_modulus = modulus
            changed.append(f"{name}={modulus}")

        if not changed:
            return []
        return [
            f"evaluation cadence reconfigured: {', '.join(changed)} "
            "(schedules reset)"
        ]

    def _apply_optimizer_config_patch(
        self,
        patch: _TrainingParameterPatch,
    ) -> None:
        optimizer_config = self.config.optimizer_config
        if patch.beta1 is not None:
            optimizer_config.beta1 = patch.beta1
        if patch.beta2 is not None:
            optimizer_config.beta2 = patch.beta2
        if patch.weight_decay is not None:
            optimizer_config.weight_decay = patch.weight_decay

        if (
            patch.beta1 is not None
            or patch.beta2 is not None
            or patch.weight_decay is not None
        ):
            self.apply_optimizer_config()

    def _apply_lr_config(self, patch: _TrainingParameterPatch) -> None:
        self.lr_schedule.update_params(
            start_lr=patch.start_lr,
            end_lr=patch.end_lr,
            max_lr=patch.max_lr,
            total_units=self.config.trainer_config.max_training_units,
            warmup_length_ratio=patch.warmup_length_ratio,
            cos_decay_start_ratio=patch.cos_decay_start_ratio,
        )
