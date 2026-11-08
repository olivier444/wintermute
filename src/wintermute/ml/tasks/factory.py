from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import torch
import torch.nn as nn

from wintermute.data.iterate.dataclasses import TrainingExample, TrainingViewConfig
from wintermute.ml.training.task_specification import TaskSpecification
from wintermute.ml.training.view import (
    CompositeSourceExhaustionObserver,
    TrainingViewIterationReport,
)
from wintermute.tools.misc import resolve_class_from_package

if TYPE_CHECKING:
    from wintermute.ml.inference.conversation import InferenceSession, InferenceTurn
    from wintermute.ml.tasks.baseline import BaselineSuite
    from wintermute.ml.training.logger import Logger
    from wintermute.ml.training.monitor import TrainingMonitor


class TaskWrapperFactory:
    def build_wrapper(
        self,
        specification: TaskSpecification,
        device: torch.device,
        bf_16_enabled: bool,
        root_path: str | Path,
    ) -> TaskWrapper:
        wrapper_cls = resolve_class_from_package(
            package_name="wintermute.ml.tasks.implementations",
            class_name=specification.wrapper_class,
            expected_base_class=TaskWrapper,
        )

        try:
            return wrapper_cls(
                device=device,
                bf_16_enabled=bf_16_enabled,
                params=specification.params,
                root_path=str(root_path),
            )
        except TypeError as exc:
            raise TypeError(
                f"Builder '{specification.wrapper_class}' has an invalid signature."
            ) from exc


class TaskWrapper(ABC):
    """Task-specific services exposed to training strategies."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        examples: list[TrainingExample],
    ) -> torch.Tensor: ...

    @abstractmethod
    def build_training_view(
        self,
        output_root: str,
        training_view_config: TrainingViewConfig,
        silent: bool = False,
        source_exhaustion_observer: CompositeSourceExhaustionObserver | None = None,
    ) -> Iterable[TrainingExample]: ...

    @abstractmethod
    def compute_eval_metrics(
        self,
        model: nn.Module,
        batches: Iterable[list[TrainingExample]],
    ) -> dict[str, float]: ...

    def count_training_units(self, batch: list[TrainingExample]) -> int:
        return sum(example.effective_units() for example in batch)

    def training_view_iteration_report(
        self,
        view: Iterable[TrainingExample],
    ) -> TrainingViewIterationReport | None:
        del view
        return None

    def start_inference_session(
        self,
        model: nn.Module,
        system_prompt: str | None = None,
        initial_turns: tuple[InferenceTurn, ...] = (),
    ) -> InferenceSession:
        del model, system_prompt, initial_turns
        raise NotImplementedError(
            f"{type(self).__name__} does not support conversational inference"
        )

    def build_baseline_suite(self, baseline_config: Mapping[str, Any]) -> BaselineSuite:
        del baseline_config
        from wintermute.ml.tasks.baseline import NoBaselineSuite

        return NoBaselineSuite()

    def build_training_monitor(
        self,
        *,
        model: nn.Module,
        logger: Logger,
        configs: Iterable[Mapping[str, Any]],
    ) -> TrainingMonitor:
        del model, logger
        if list(configs):
            raise ValueError(
                f"{type(self).__name__} does not support training monitors"
            )

        from wintermute.ml.training.monitor import NoTrainingMonitor

        return NoTrainingMonitor()

    def export_assets(self, output_dir: str | Path) -> None:
        del output_dir
