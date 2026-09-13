from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Deque, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer

from wintermute.data.iterate.dataclasses import TrainingExample
from wintermute.ml.tasks.factory import TaskWrapper
from wintermute.tools.model import preserving_torch_rng_state


@dataclass(frozen=True)
class RecordedMicroBatch:
    examples: List[TrainingExample]
    units: int
    grad_accum_size_units: int
    loss_before_update: float
    cpu_rng_state: Optional[Tensor]
    cuda_rng_state: Optional[List[Tensor]]

    @property
    def gradient_weight(self) -> float:
        return self.units / self.grad_accum_size_units


@dataclass(frozen=True)
class RecordedUpdateBatch:
    micro_batches: List[RecordedMicroBatch]
    total_units: int


@dataclass(frozen=True)
class GradientNoiseEstimate:
    update_batch_count: int
    mean_squared_distance: float
    mean_grad_norm: float
    mean_update_batch_grad_norm: float
    avg_update_batch_units: float
    total_units: int
    elapsed_sec: float

    def noise_ratio(self) -> float:
        return math.sqrt(self.mean_squared_distance) / self.mean_grad_norm

    def critical_batch_size_units(self) -> float:
        if self.update_batch_count <= 1 or self.mean_grad_norm <= 0:
            return float("nan")

        sample_correction = self.update_batch_count / (self.update_batch_count - 1)
        return (
            self.avg_update_batch_units
            * sample_correction
            * self.mean_squared_distance
            / (self.mean_grad_norm * self.mean_grad_norm)
        )


@dataclass(frozen=True)
class FirstUpdateDiagnostics:
    grad_norm: float
    param_norm: float
    delta_param_norm: float
    neg_grad_update_cos: float
    grad_dot_update: float
    observed_loss_decrease: float
    observed_to_predicted_loss_decrease_ratio: float

    @property
    def delta_param_to_param(self) -> float:
        if self.param_norm <= 0:
            return 0.0
        return self.delta_param_norm / self.param_norm

    @property
    def grad_to_param(self) -> float:
        if self.param_norm <= 0:
            return 0.0
        return self.grad_norm / self.param_norm

    @classmethod
    def from_snapshot(cls, snapshot: UpdateSnapshot) -> FirstUpdateDiagnostics:
        return cls(
            grad_norm=snapshot.grad_norm,
            param_norm=snapshot.param_norm,
            delta_param_norm=snapshot.delta_param_norm,
            neg_grad_update_cos=snapshot.neg_grad_update_cos,
            grad_dot_update=snapshot.grad_dot_update,
            observed_loss_decrease=snapshot.observed_loss_decrease,
            observed_to_predicted_loss_decrease_ratio=(
                snapshot.observed_to_predicted_loss_decrease_ratio
            ),
        )


@dataclass(frozen=True)
class UpdatePairDiagnostics:
    grad_cos_prev: float
    update_cos_prev: float


@dataclass(frozen=True)
class PendingUpdateSnapshot:
    param_norm: float
    cpu_param_snapshot: List[Tensor]
    cpu_grad_snapshot: List[Tensor]


@dataclass(frozen=True)
class UpdateSnapshot:
    grad_norm: float
    param_norm: float
    delta_param_norm: float
    neg_grad_update_cos: float
    grad_dot_update: float
    observed_loss_decrease: float
    observed_to_predicted_loss_decrease_ratio: float
    cpu_grad_snapshot: List[Tensor]
    cpu_update_snapshot: List[Tensor]


@dataclass(frozen=True)
class OptimizerDiagnosticsResult:
    last_grad_norm: float
    adam_sqrt_v_rms: float
    adam_m_over_sqrt_v_rms: float
    first_update: Optional[FirstUpdateDiagnostics]
    update_pair: Optional[UpdatePairDiagnostics]
    grad_clip_fraction: float
    optimizer_step_count: int
    gradient_noise_estimate: Optional[GradientNoiseEstimate]
    gradient_noise_available_update_batches: int
    gradient_noise_required_update_batches: int
    gradient_noise_estimation_due: bool
    gradient_noise_modulus: int


class OptimizerDiagnosticsTracker:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        task_wrapper: TaskWrapper,
        gradient_noise_update_batch_count: int,
        gradient_noise_modulus: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.task_wrapper = task_wrapper
        self._pending_micro_batches: List[RecordedMicroBatch] = []
        self._pending_units = 0
        self._recent_update_batches: Deque[RecordedUpdateBatch] = deque()
        self._gradient_noise_update_batch_count = 0
        self._gradient_noise_modulus = 1
        self._macro_steps_since_gradient_noise_estimate = 0
        self._gradient_noise_estimate: Optional[GradientNoiseEstimate] = None
        self._gradient_noise_estimation_due = False
        self.reconfigure_gradient_noise_update_batch_count(
            gradient_noise_update_batch_count
        )
        self.reconfigure_gradient_noise_modulus(gradient_noise_modulus)
        self.reset_macro_step()

    @property
    def gradient_noise_update_batch_count(self) -> int:
        return self._gradient_noise_update_batch_count

    @property
    def gradient_noise_modulus(self) -> int:
        return self._gradient_noise_modulus

    def reset_macro_step(self) -> None:
        self._optimizer_step_count = 0
        self._grad_clip_count = 0
        self._first_update_snapshot: Optional[UpdateSnapshot] = None
        self._first_update_diagnostics: Optional[FirstUpdateDiagnostics] = None
        self._update_pair_diagnostics: Optional[UpdatePairDiagnostics] = None
        self._last_grad_norm = 0.0
        self._adam_sqrt_v_rms = float("nan")
        self._adam_m_over_sqrt_v_rms = float("nan")
        self._gradient_noise_estimate = None
        self._gradient_noise_estimation_due = False

    def reconfigure_gradient_noise_update_batch_count(self, count: int) -> bool:
        normalized_count = max(0, int(count))
        changed = normalized_count != self._gradient_noise_update_batch_count
        self._gradient_noise_update_batch_count = normalized_count
        self._recent_update_batches = deque(maxlen=max(1, normalized_count))
        self._pending_micro_batches = []
        self._pending_units = 0
        self._gradient_noise_estimate = None
        self._gradient_noise_estimation_due = False
        return changed

    def reconfigure_gradient_noise_modulus(self, modulus: int) -> bool:
        normalized_modulus = max(1, int(modulus))
        if normalized_modulus == self._gradient_noise_modulus:
            return False

        self._gradient_noise_modulus = normalized_modulus
        self._macro_steps_since_gradient_noise_estimate = 0
        self._gradient_noise_estimate = None
        self._gradient_noise_estimation_due = False
        return True

    def should_capture_observed_loss(self) -> bool:
        return self._optimizer_step_count == 0

    def record_micro_batch(
        self,
        *,
        batch: List[TrainingExample],
        batch_units: int,
        grad_accum_size_units: int,
        loss_before_update: float,
        cpu_rng_state: Optional[Tensor],
        cuda_rng_state: Optional[List[Tensor]],
    ) -> None:
        if batch_units <= 0:
            return
        if (
            self._gradient_noise_update_batch_count <= 0
            and not self.should_capture_observed_loss()
        ):
            return

        self._pending_micro_batches.append(
            RecordedMicroBatch(
                examples=list(batch),
                units=int(batch_units),
                grad_accum_size_units=grad_accum_size_units,
                loss_before_update=loss_before_update,
                cpu_rng_state=cpu_rng_state,
                cuda_rng_state=cuda_rng_state,
            )
        )
        self._pending_units += int(batch_units)

    def prepare_optimizer_step(self) -> Optional[PendingUpdateSnapshot]:
        if self._optimizer_step_count >= 2:
            return None
        return self._capture_pending_update_snapshot()

    def complete_optimizer_step(
        self,
        pending_snapshot: Optional[PendingUpdateSnapshot],
        *,
        raw_grad_norm: float,
        grad_clip: float,
    ) -> None:
        optimizer_step_index = self._optimizer_step_count
        (
            self._adam_sqrt_v_rms,
            self._adam_m_over_sqrt_v_rms,
        ) = self._compute_adam_moment_rms()

        observed_loss_decrease = (
            self._compute_observed_loss_decrease()
            if optimizer_step_index == 0
            else float("nan")
        )
        if optimizer_step_index < 2:
            assert pending_snapshot is not None
            update_snapshot = self._complete_update_snapshot(
                pending_snapshot,
                raw_grad_norm,
                observed_loss_decrease,
            )
            if optimizer_step_index == 0:
                self._first_update_snapshot = update_snapshot
                self._first_update_diagnostics = FirstUpdateDiagnostics.from_snapshot(
                    update_snapshot
                )
            elif self._first_update_snapshot is not None:
                self._update_pair_diagnostics = self._compute_update_pair_diagnostics(
                    self._first_update_snapshot,
                    update_snapshot,
                )

        self._last_grad_norm = raw_grad_norm
        self._optimizer_step_count += 1
        if raw_grad_norm >= grad_clip:
            self._grad_clip_count += 1
        self._record_completed_update_batch()

    def estimate_gradient_noise_if_due(self) -> None:
        self._gradient_noise_estimate = None
        self._gradient_noise_estimation_due = False
        if self._gradient_noise_update_batch_count <= 0:
            return

        self._macro_steps_since_gradient_noise_estimate += 1
        if (
            self._macro_steps_since_gradient_noise_estimate
            < self._gradient_noise_modulus
        ):
            return

        self._macro_steps_since_gradient_noise_estimate = 0
        self._gradient_noise_estimation_due = True
        self._gradient_noise_estimate = self._compute_gradient_noise()

    def summary(self) -> OptimizerDiagnosticsResult:
        grad_clip_fraction = (
            self._grad_clip_count / self._optimizer_step_count
            if self._optimizer_step_count > 0
            else 0.0
        )
        available_update_batches = (
            len(self._recent_update_batches)
            if self._gradient_noise_update_batch_count > 0
            else 0
        )
        return OptimizerDiagnosticsResult(
            last_grad_norm=self._last_grad_norm,
            adam_sqrt_v_rms=self._adam_sqrt_v_rms,
            adam_m_over_sqrt_v_rms=self._adam_m_over_sqrt_v_rms,
            first_update=self._first_update_diagnostics,
            update_pair=self._update_pair_diagnostics,
            grad_clip_fraction=grad_clip_fraction,
            optimizer_step_count=self._optimizer_step_count,
            gradient_noise_estimate=self._gradient_noise_estimate,
            gradient_noise_available_update_batches=available_update_batches,
            gradient_noise_required_update_batches=(
                self._gradient_noise_update_batch_count
            ),
            gradient_noise_estimation_due=self._gradient_noise_estimation_due,
            gradient_noise_modulus=self._gradient_noise_modulus,
        )

    def _record_completed_update_batch(self) -> None:
        if (
            self._gradient_noise_update_batch_count > 0
            and self._pending_units > 0
            and self._pending_micro_batches
        ):
            self._recent_update_batches.append(
                RecordedUpdateBatch(
                    micro_batches=list(self._pending_micro_batches),
                    total_units=self._pending_units,
                )
            )
        self._pending_micro_batches = []
        self._pending_units = 0

    def _compute_gradient_noise(self) -> Optional[GradientNoiseEstimate]:
        update_batches = list(self._recent_update_batches)
        if len(update_batches) < self._gradient_noise_update_batch_count:
            return None

        started_at = time.perf_counter()
        trainable_params = [
            param for param in self.model.parameters() if param.requires_grad
        ]
        grad_sums = [
            torch.zeros(param.shape, dtype=torch.float32, device="cpu")
            for param in trainable_params
        ]
        total_grad_sq = 0.0
        total_grad_norm = 0.0

        with preserving_torch_rng_state():
            self.model.zero_grad(set_to_none=True)
            try:
                for update_batch in update_batches:
                    self.model.zero_grad(set_to_none=True)
                    for micro_batch in update_batch.micro_batches:
                        loss = self.task_wrapper.compute_loss(
                            self.model,
                            micro_batch.examples,
                        )
                        (loss * micro_batch.gradient_weight).backward()

                    batch_grad_sq = 0.0
                    for index, param in enumerate(trainable_params):
                        grad = param.grad
                        if grad is None:
                            continue
                        grad_cpu = grad.detach().to(
                            device="cpu",
                            dtype=torch.float32,
                        )
                        grad_sums[index].add_(grad_cpu)
                        batch_grad_sq += float(torch.sum(grad_cpu * grad_cpu).item())

                    total_grad_sq += batch_grad_sq
                    total_grad_norm += math.sqrt(batch_grad_sq)
            finally:
                self.model.zero_grad(set_to_none=True)

        batch_count = len(update_batches)
        mean_squared_grad_norm = total_grad_sq / batch_count
        mean_grad_squared_norm = sum(
            float(torch.sum(grad_sum * grad_sum).item())
            for grad_sum in grad_sums
        ) / float(batch_count * batch_count)
        total_units = sum(batch.total_units for batch in update_batches)
        return GradientNoiseEstimate(
            update_batch_count=batch_count,
            mean_squared_distance=max(
                0.0,
                mean_squared_grad_norm - mean_grad_squared_norm,
            ),
            mean_grad_norm=math.sqrt(max(0.0, mean_grad_squared_norm)),
            mean_update_batch_grad_norm=total_grad_norm / batch_count,
            avg_update_batch_units=total_units / batch_count,
            total_units=total_units,
            elapsed_sec=time.perf_counter() - started_at,
        )

    def _compute_param_norm(self) -> float:
        total_sq: Optional[Tensor] = None
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            param_sq = torch.sum(param.detach() ** 2)
            total_sq = param_sq if total_sq is None else total_sq + param_sq

        if total_sq is None:
            return 0.0
        return math.sqrt(float(total_sq.item()))

    def _snapshot_trainable_params_to_cpu(self) -> List[Tensor]:
        return [
            param.detach().to(device="cpu", dtype=torch.float32)
            for param in self.model.parameters()
            if param.requires_grad
        ]

    def _snapshot_trainable_grads_to_cpu(self) -> List[Tensor]:
        cpu_snapshot: List[Tensor] = []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                cpu_snapshot.append(
                    torch.zeros_like(
                        param.detach(),
                        device="cpu",
                        dtype=torch.float32,
                    )
                )
            else:
                cpu_snapshot.append(
                    param.grad.detach().to(device="cpu", dtype=torch.float32)
                )
        return cpu_snapshot

    def _capture_pending_update_snapshot(self) -> PendingUpdateSnapshot:
        return PendingUpdateSnapshot(
            param_norm=self._compute_param_norm(),
            cpu_param_snapshot=self._snapshot_trainable_params_to_cpu(),
            cpu_grad_snapshot=self._snapshot_trainable_grads_to_cpu(),
        )

    def _compute_param_delta_from_cpu_snapshot(
        self,
        cpu_snapshot: List[Tensor],
    ) -> Tuple[List[Tensor], float]:
        deltas = [
            param.detach().to(device="cpu", dtype=torch.float32) - previous
            for param, previous in zip(
                (param for param in self.model.parameters() if param.requires_grad),
                cpu_snapshot,
            )
        ]
        total_sq = sum(float(torch.sum(delta**2).item()) for delta in deltas)
        return deltas, math.sqrt(total_sq)

    @staticmethod
    def _dot_and_cosine_between_cpu_vectors(
        left: List[Tensor],
        right: List[Tensor],
    ) -> Tuple[float, float]:
        dot = 0.0
        left_sq = 0.0
        right_sq = 0.0
        for x, y in zip(left, right):
            dot += float(torch.sum(x * y).item())
            left_sq += float(torch.sum(x**2).item())
            right_sq += float(torch.sum(y**2).item())

        denominator = math.sqrt(left_sq * right_sq)
        cosine = dot / denominator if denominator > 0 else float("nan")
        return dot, cosine

    def _complete_update_snapshot(
        self,
        pending_snapshot: PendingUpdateSnapshot,
        grad_norm: float,
        observed_loss_decrease: float,
    ) -> UpdateSnapshot:
        cpu_update_snapshot, delta_param_norm = (
            self._compute_param_delta_from_cpu_snapshot(
                pending_snapshot.cpu_param_snapshot
            )
        )
        grad_dot_update, grad_update_cos = self._dot_and_cosine_between_cpu_vectors(
            pending_snapshot.cpu_grad_snapshot,
            cpu_update_snapshot,
        )
        predicted_loss_decrease = -grad_dot_update
        observed_to_predicted_loss_decrease_ratio = (
            observed_loss_decrease / predicted_loss_decrease
            if predicted_loss_decrease != 0
            else float("nan")
        )
        return UpdateSnapshot(
            grad_norm=grad_norm,
            param_norm=pending_snapshot.param_norm,
            delta_param_norm=delta_param_norm,
            neg_grad_update_cos=-grad_update_cos,
            grad_dot_update=grad_dot_update,
            observed_loss_decrease=observed_loss_decrease,
            observed_to_predicted_loss_decrease_ratio=(
                observed_to_predicted_loss_decrease_ratio
            ),
            cpu_grad_snapshot=pending_snapshot.cpu_grad_snapshot,
            cpu_update_snapshot=cpu_update_snapshot,
        )

    def _compute_observed_loss_decrease(self) -> float:
        micro_batches = self._pending_micro_batches
        loss_before_update = sum(
            micro_batch.gradient_weight * micro_batch.loss_before_update
            for micro_batch in micro_batches
        )
        loss_after_update = 0.0
        with preserving_torch_rng_state(), torch.no_grad():
            for micro_batch in micro_batches:
                assert micro_batch.cpu_rng_state is not None
                torch.set_rng_state(micro_batch.cpu_rng_state)
                if micro_batch.cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(micro_batch.cuda_rng_state)
                loss_after_update += micro_batch.gradient_weight * float(
                    self.task_wrapper.compute_loss(
                        self.model,
                        micro_batch.examples,
                    )
                    .detach()
                    .item()
                )
        return loss_before_update - loss_after_update

    def _compute_adam_moment_rms(self) -> Tuple[float, float]:
        sum_v: Optional[Tensor] = None
        sum_m_over_sqrt_v_sq: Optional[Tensor] = None
        element_count = 0
        for state in self.optimizer.state.values():
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            if not isinstance(exp_avg, Tensor) or not isinstance(exp_avg_sq, Tensor):
                continue

            detached_exp_avg = exp_avg.detach()
            detached_exp_avg_sq = exp_avg_sq.detach()
            sum_v_term = torch.sum(detached_exp_avg_sq)
            min_positive_v = torch.finfo(detached_exp_avg_sq.dtype).tiny
            m_over_sqrt_v_sq_term = torch.sum(
                detached_exp_avg.square()
                / detached_exp_avg_sq.clamp_min(min_positive_v)
            )
            sum_v = sum_v_term if sum_v is None else sum_v + sum_v_term
            sum_m_over_sqrt_v_sq = (
                m_over_sqrt_v_sq_term
                if sum_m_over_sqrt_v_sq is None
                else sum_m_over_sqrt_v_sq + m_over_sqrt_v_sq_term
            )
            element_count += exp_avg_sq.numel()

        if sum_v is None or sum_m_over_sqrt_v_sq is None or element_count == 0:
            return float("nan"), float("nan")
        return (
            float(torch.sqrt(sum_v / element_count).item()),
            float(torch.sqrt(sum_m_over_sqrt_v_sq / element_count).item()),
        )

    def _compute_update_pair_diagnostics(
        self,
        previous: UpdateSnapshot,
        current: UpdateSnapshot,
    ) -> UpdatePairDiagnostics:
        _, grad_cos_prev = self._dot_and_cosine_between_cpu_vectors(
            current.cpu_grad_snapshot,
            previous.cpu_grad_snapshot,
        )
        _, update_cos_prev = self._dot_and_cosine_between_cpu_vectors(
            current.cpu_update_snapshot,
            previous.cpu_update_snapshot,
        )
        return UpdatePairDiagnostics(
            grad_cos_prev=grad_cos_prev,
            update_cos_prev=update_cos_prev,
        )
