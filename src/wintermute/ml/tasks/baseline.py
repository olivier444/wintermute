from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Dict

import torch.nn as nn

from wintermute.data.iterate.dataclasses import TrainingViewConfig
from wintermute.ml.tasks.factory import TaskWrapper
from wintermute.tools.logging import console_log
from wintermute.tools.model import evaluating, unload_from_gpu, iter_batches


class BaselineSuite(ABC):
    @abstractmethod
    def compute_eval_metrics(
        self,
        view_config: TrainingViewConfig
    ) -> Dict[str, Dict[str, float]]: ...

    def benchmark_evaluators(self) -> Sequence[BaselineEvaluator]:
        return ()


class NoBaselineSuite(BaselineSuite):
    def compute_eval_metrics(
        self,
        view_config: TrainingViewConfig
    ) -> Dict[str, Dict[str, float]]:
        return {}


class BaselineEvaluator:
    def __init__(
            self, 
            name: str, 
            model: nn.Module, 
            task_wrapper: TaskWrapper,
            batch_size: int
        ) -> None:

        self.name = name
        self.task_wrapper = task_wrapper
        self.model = model
        self.batch_size = batch_size


    def compute_eval_metrics(
        self,
        view_config: TrainingViewConfig,
        output_root: str
    ) -> Dict[str, float]:
        self.model = self.model.to(self.task_wrapper.device)

        console_log("baseline", f"computing eval metrics for {self.name} ...")

        view = self.task_wrapper.build_training_view(
            output_root=output_root,
            training_view_config=view_config,
            silent=True,
        )
        
        with evaluating(self.model):
            batches = iter_batches(view, self.batch_size)
            result = self.task_wrapper.compute_eval_metrics(self.model, batches)
            self.model = unload_from_gpu(self.model)
            return result
