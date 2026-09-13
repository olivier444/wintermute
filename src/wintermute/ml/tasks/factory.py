from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import TYPE_CHECKING, Any, List, Dict, Iterable, Mapping
import torch
import torch.nn as nn
from wintermute.data.iterate.dataclasses import TrainingExample, TrainingViewConfig
from wintermute.ml.tokenization.chat_format import AssistantCompletion
from wintermute.ml.training.task_specification import TaskSpecification

from wintermute.tools.misc import resolve_class_from_package
from wintermute.tools.slack.text import format_slack_code_block

if TYPE_CHECKING:
    from wintermute.ml.tasks.baseline import BaselineSuite
    from wintermute.ml.training.logger import Logger
    from wintermute.ml.training.monitor import TrainingMonitor


@dataclass(frozen=True)
class OutputPreview:
    payload: object

    def as_text_list(self) -> List[str]:
        if isinstance(self.payload, list):
            return self.payload
        
        return [self.as_text()]

    def as_text(self) -> str:
        if isinstance(self.payload, AssistantCompletion):
            return self.payload.final

        if isinstance(self.payload, str):
            return self.payload
        
        if isinstance(self.payload, list):
            return "\n--------------------------------\n".join(self.payload)     

        return json.dumps(self.payload, ensure_ascii=False, indent=2, default=str)


@dataclass
class InferenceSettings:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.85
    top_k: int | None = 50
    greedy: bool = False
    force_final: bool = False

    def clone(self) -> "InferenceSettings":
        return InferenceSettings(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            greedy=self.greedy,
            force_final=self.force_final,
        )

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be > 0")

    def describe(
        self,
        *,
        history_turn_count: int | None = None,
        chat_format_enabled: bool | None = None,
    ) -> str:
        self.validate()
        top_k = "off" if self.top_k is None else str(self.top_k)
        mode = "greedy" if self.greedy else "sample"
        setting_lines = [
            f"mode={mode}",
            f"max_tokens={self.max_new_tokens}",
            f"temperature={self.temperature:.3f}",
            f"top_p={self.top_p:.3f}",
            f"top_k={top_k}",
            f"force_final={'on' if self.force_final else 'off'}",
        ]
        if chat_format_enabled is not None:
            setting_lines.append(f"chat_format={'on' if chat_format_enabled else 'off'}")
        if history_turn_count is not None:
            setting_lines.append(f"history_turns={history_turn_count}")
        return "[info] inference settings\n" + format_slack_code_block("\n".join(setting_lines))


@dataclass(frozen=True)
class InferenceTurnState:
    user_text: str
    assistant_text: str


@dataclass(frozen=True)
class InferenceSessionState:
    turns: tuple[InferenceTurnState, ...] = ()
    settings: InferenceSettings = field(default_factory=InferenceSettings)
    use_chat_format: bool = True


class InferenceSession(ABC):
    @property
    @abstractmethod
    def settings(self) -> InferenceSettings: ...

    @abstractmethod
    def infer(self, arguments: str) -> OutputPreview: (...)

    def reset(self) -> None:
        raise NotImplementedError("This inference session does not support reset()")

    def export_state(self) -> InferenceSessionState:
        raise NotImplementedError("This inference session does not support export_state()")

    @property
    def history_turn_count(self) -> int | None:
        return None


class TaskWrapperFactory:
    def __init__(self) -> None:
        return
    

    def build_wrapper(
            self,
            specification: TaskSpecification,
            device: torch.device, 
            bf_16_enabled: bool,
            root_path: str | Path,
        ) -> TaskWrapper:

        builder_cls = resolve_class_from_package(
            package_name="wintermute.ml.tasks.implementations",
            class_name=specification.wrapper_class,
            expected_base_class=TaskWrapper
        )

        try:
            return builder_cls(
                device=device,
                bf_16_enabled=bf_16_enabled,
                params=specification.params,
                root_path=str(root_path),
            )
        except TypeError as exc:
            raise TypeError(f"Builder '{specification.wrapper_class}' has an invalid signature.") from exc        


class TaskWrapper(ABC):
    def __init__(self, device: torch.device):
        self.device = device
        return
    

    @abstractmethod
    def compute_loss(
            self, 
            model: nn.Module, 
            examples: List[TrainingExample]
        ) -> torch.Tensor: (...)


    @abstractmethod
    def build_training_view(
            self,
            output_root: str,
            training_view_config: TrainingViewConfig,
            silent: bool = False
        ) -> Iterable[TrainingExample]: (...)
    

    @abstractmethod
    def compute_eval_metrics(
            self, 
            model: nn.Module, 
            batches: Iterable[List[TrainingExample]]
        ) -> Dict[str, float]: (...)


    @abstractmethod
    def start_inference_session(
        self,
        model: nn.Module,
        system_prompt: str | None = None,
        initial_state: InferenceSessionState | None = None,
    ) -> InferenceSession: (...)


    def build_baseline_suite(self, baseline_config: Mapping[str, Any]) -> BaselineSuite:
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
        return
