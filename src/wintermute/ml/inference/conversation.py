from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

from wintermute.ml.tokenization.chat_format import AssistantCompletion


@dataclass(frozen=True)
class OutputPreview:
    payload: object

    def as_text_list(self) -> list[str]:
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


@dataclass(frozen=True)
class InferenceSettings:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.85
    top_k: int | None = 50
    greedy: bool = False
    force_final: bool = False
    use_chat_format: bool = True

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
    ) -> "InferenceSettingsDescription":
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
            f"chat_format={'on' if self.use_chat_format else 'off'}",
        ]
        if history_turn_count is not None:
            setting_lines.append(f"history_turns={history_turn_count}")
        return InferenceSettingsDescription(lines=tuple(setting_lines))


@dataclass(frozen=True)
class InferenceSettingsDescription:
    lines: tuple[str, ...]

    def as_text(self) -> str:
        return "[info] inference settings\n" + "\n".join(self.lines)


@dataclass(frozen=True)
class InferenceTurn:
    user_text: str
    assistant_text: str


class InferenceSession(ABC):
    @abstractmethod
    def infer(self, arguments: str, settings: InferenceSettings) -> OutputPreview: ...

    def reset(self) -> None:
        raise NotImplementedError("This inference session does not support reset()")

    def export_turns(self) -> tuple[InferenceTurn, ...]:
        raise NotImplementedError("This inference session does not support export_turns()")

    @property
    def history_turn_count(self) -> int | None:
        return None
