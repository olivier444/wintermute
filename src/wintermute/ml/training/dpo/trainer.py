from __future__ import annotations

import torch
import torch.nn as nn

from wintermute.ml.models.base import BaseModel
from wintermute.ml.tasks.factory import TaskWrapper
from wintermute.ml.training.config import DpoTrainerConfig, RunConfig
from wintermute.ml.training.trainer import Trainer


class DpoTrainer(Trainer):
    """Online DPO trainer scaffold."""

    def __init__(
        self,
        *,
        model: BaseModel,
        task_wrapper: TaskWrapper,
        config: RunConfig,
        device: torch.device,
    ) -> None:
        dpo_config = config.trainer_config
        if not isinstance(dpo_config, DpoTrainerConfig):
            raise TypeError("DpoTrainer requires a DPO trainer configuration")

        self.model = model
        self.task_wrapper = task_wrapper
        self.config = config
        self.dpo_config = dpo_config
        self.device = device

        self.model.to(self.device)
        self.model.train()

    def train(self) -> nn.Module:
        return self.model
