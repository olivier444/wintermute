from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class BurnContext:
    started_at: float = field(init=False)
    last_log_at: float = field(init=False)

    def __post_init__(self) -> None:
        now = time.perf_counter()
        self.started_at = now
        self.last_log_at = now


@dataclass
class MacroStepContext:
    sum_train_loss_weighted: float = 0.0
    sum_train_units: int = 0
    eval_checkpoint_sec: float = 0.0
    eval_start_time: float = 0.0
    monitor_checkpoint_sec: float = 0.0
    monitor_start_time: float = 0.0
    perf_window_started_at: float = field(init=False)
    perf_window_units0: int = 0
    perf_window_gs0: int = 0

    def __post_init__(self) -> None:
        self.perf_window_started_at = time.perf_counter()

    def reset(self) -> None:
        self.sum_train_loss_weighted = 0.0
        self.sum_train_units = 0
        self.eval_checkpoint_sec = 0.0
        self.eval_start_time = 0.0
        self.monitor_checkpoint_sec = 0.0
        self.monitor_start_time = 0.0

    def on_eval_started(self) -> None:
        self.eval_start_time = time.perf_counter()

    def on_eval_completed(self) -> None:
        self.eval_checkpoint_sec = time.perf_counter() - self.eval_start_time

    def on_monitor_started(self) -> None:
        self.monitor_start_time = time.perf_counter()

    def on_monitor_completed(self) -> None:
        self.monitor_checkpoint_sec = time.perf_counter() - self.monitor_start_time

    def on_micro_batch(self, last_train_loss: float, batch_units: int) -> None:
        self.sum_train_loss_weighted += last_train_loss * batch_units
        self.sum_train_units += batch_units

    def compute_macro_train_loss(self) -> float:
        if self.sum_train_units <= 0:
            raise RuntimeError(
                "Cannot compute macro train loss without any supervised units."
            )
        return self.sum_train_loss_weighted / self.sum_train_units

    def on_macro_step_completed(self, *, units_seen: int, global_step: int) -> None:
        self.perf_window_started_at = time.perf_counter()
        self.perf_window_units0 = units_seen
        self.perf_window_gs0 = global_step

    def compute_perf_metrics(
        self,
        *,
        units_seen: int,
        global_step: int,
    ) -> Tuple[float, float]:
        elapsed_sec = time.perf_counter() - self.perf_window_started_at
        units_per_sec = (
            (units_seen - self.perf_window_units0) / elapsed_sec
            if elapsed_sec > 0
            else 0.0
        )
        macro_steps = global_step - self.perf_window_gs0
        sec_per_step = elapsed_sec / macro_steps if macro_steps > 0 else 0.0
        return units_per_sec, sec_per_step
