from wintermute.ml.training.config import LrConfig
import math
from typing import Optional

from wintermute.tools.logging import console_log


class LrSchedule:
    def __init__(self, config: LrConfig):
        self.config = config
        self.total_units = 0
        self.current_lr = 0


    def update_params(
            self, 
            *,
            total_units: Optional[int] = None, 
            start_lr: Optional[float] = None,
            end_lr: Optional[float] = None,
            max_lr: Optional[float] = None,
            warmup_length_ratio: Optional[float] = None,
            cos_decay_start_ratio: Optional[float] = None,
        ):

        console_log(
            "lr_schedule",
            "updating params - "
            f"start_lr:{start_lr} - end_lr:{end_lr} - max_lr:{max_lr} - total_units:{total_units} - "
            f"warmup_length_ratio:{warmup_length_ratio} - cos_decay_start_ratio:{cos_decay_start_ratio}",
        )

        if total_units is not None:
            self.total_units = total_units

        if start_lr is not None:
            self.config.start_lr = start_lr

        if end_lr is not None:
            self.config.end_lr = end_lr

        if max_lr is not None:
            self.config.max_lr = max_lr

        if warmup_length_ratio is not None:
            self.config.warmup_length_ratio = warmup_length_ratio

        if cos_decay_start_ratio is not None:
            self.config.cos_decay_start_ratio = cos_decay_start_ratio

        self.warmup_length_units: int = int(max(1, self.total_units * self.config.warmup_length_ratio))
        self.cos_decay_start_units: int = int(max(self.warmup_length_units, self.total_units * self.config.cos_decay_start_ratio))
        self.cos_duration_units = self.total_units - self.cos_decay_start_units
        self.update_lr(0)
        
    
    def update_lr(self, units_seen: int) -> None:
        if units_seen < self.warmup_length_units:
            warmup_progress = float(units_seen) / float(self.warmup_length_units)
            self.current_lr = self.config.start_lr + (self.config.max_lr - self.config.start_lr) * warmup_progress
        elif units_seen < self.cos_decay_start_units:
            self.current_lr = self.config.max_lr 
        elif self.cos_duration_units <= 0:
            self.current_lr = self.config.end_lr
        else:
            p = (units_seen - self.cos_decay_start_units) / float(self.cos_duration_units)
            p = min(1.0, max(0.0, p))
            cosine = 0.5 * (1.0 + math.cos(math.pi * p))
            self.current_lr = self.config.end_lr + (self.config.max_lr - self.config.end_lr) * cosine
