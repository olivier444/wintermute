from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, cast

from transformers import PreTrainedTokenizerBase


ASSISTANT_FORMAT_LEGACY = "legacy"
ASSISTANT_FORMAT_NORMALIZED = "normalized"
_ALLOWED_ASSISTANT_FORMATS = {ASSISTANT_FORMAT_LEGACY, ASSISTANT_FORMAT_NORMALIZED}

WINTERMUTE_SYSTEM = "[:system:]"
WINTERMUTE_USER = "[:user:]"
WINTERMUTE_ASSISTANT = "[:assistant:]"
WINTERMUTE_EOT = "[:eot:]"
WINTERMUTE_TASK = "[:task:]"
WINTERMUTE_VERBOSITY = "[:verbosity:]"
WINTERMUTE_FORMAT = "[:format:]"
WINTERMUTE_SCRATCHPAD = "[:scratchpad:]"
WINTERMUTE_FINAL = "[:final:]"
WINTERMUTE_CHAT_CONTROL_LITERALS = (
    WINTERMUTE_SYSTEM,
    WINTERMUTE_USER,
    WINTERMUTE_ASSISTANT,
    WINTERMUTE_EOT,
    WINTERMUTE_TASK,
    WINTERMUTE_VERBOSITY,
    WINTERMUTE_FORMAT,
    WINTERMUTE_SCRATCHPAD,
    WINTERMUTE_FINAL,
)

THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


@dataclass(frozen=True)
class ConversationTurn:
    user: str
    assistant: str


@dataclass(frozen=True)
class AssistantCompletion:
    final: str
    task: str | None = None
    verbosity: str | None = None
    format: str | None = None
    scratchpad: str | None = None


@dataclass(frozen=True)
class ConversationContext:
    current_user: str
    history: tuple[ConversationTurn, ...] = ()
    system: str | None = None


class AssistantControlToken(str, Enum):
    ASSISTANT = "assistant"
    TASK = "task"
    VERBOSITY = "verbosity"
    FORMAT = "format"
    SCRATCHPAD = "scratchpad"
    FINAL = "final"


@dataclass(frozen=True)
class AssistantPromptFormat:
    last_prompt_control_token: AssistantControlToken = AssistantControlToken.ASSISTANT
    forced_task: str | None = None
    forced_verbosity: str | None = None
    forced_format: str | None = None

    def __post_init__(self) -> None:
        forced_values = (
            ("forced_task", self.forced_task, AssistantControlToken.TASK),
            ("forced_verbosity", self.forced_verbosity, AssistantControlToken.VERBOSITY),
            ("forced_format", self.forced_format, AssistantControlToken.FORMAT),
        )
        for field_name, value, required_token in forced_values:
            if value is None:
                continue
            if not value:
                raise ValueError(f"AssistantPromptFormat.{field_name} must not be empty")
            if not self._includes(required_token):
                raise ValueError(
                    f"AssistantPromptFormat.{field_name} requires "
                    f"last_prompt_control_token to include {required_token.value}"
                )

    def _includes(self, token: AssistantControlToken) -> bool:
        token_rank = {
            AssistantControlToken.ASSISTANT: 0,
            AssistantControlToken.TASK: 1,
            AssistantControlToken.VERBOSITY: 2,
            AssistantControlToken.FORMAT: 3,
            AssistantControlToken.SCRATCHPAD: 4,
            AssistantControlToken.FINAL: 4,
        }
        return token_rank[self.last_prompt_control_token] >= token_rank[token]


@dataclass(frozen=True)
class RenderedContextBlocks:
    system: str | None
    history: tuple[str, ...]
    current: str

    @property
    def text(self) -> str:
        return "".join(part for part in (self.system, *self.history, self.current) if part)


class ChatFormat(Protocol):
    """Textual conversation protocol shared by training and inference."""

    def render_prompt(
        self,
        context: ConversationContext,
        assistant_prompt_format: AssistantPromptFormat | None = None,
    ) -> RenderedContextBlocks: ...

    def render_completion(
        self,
        *,
        final: str,
        task: str | None = None,
        verbosity: str | None = None,
        format: str | None = None,
        scratchpad: str | None = None,
        include_terminator: bool = True,
    ) -> str: ...

    def parse_completion(
        self,
        text: str,
        assistant_prompt_format: AssistantPromptFormat | None = None,
    ) -> AssistantCompletion: ...


@dataclass(frozen=True)
class WintermuteChatFormat:
    assistant_format: str = ASSISTANT_FORMAT_NORMALIZED

    def __post_init__(self) -> None:
        if self.assistant_format not in _ALLOWED_ASSISTANT_FORMATS:
            allowed = ", ".join(sorted(_ALLOWED_ASSISTANT_FORMATS))
            raise ValueError(
                f"Unsupported assistant_format '{self.assistant_format}'. Expected one of: {allowed}"
            )

    def render_prompt(
        self,
        context: ConversationContext,
        assistant_prompt_format: AssistantPromptFormat | None = None,
    ) -> RenderedContextBlocks:
        assistant_start = WINTERMUTE_ASSISTANT
        if self.uses_final_marker():
            assistant_start += (
                WINTERMUTE_FINAL
                if assistant_prompt_format is None
                else self._render_assistant_prefix(assistant_prompt_format)
            )
        return self._render_context(context, assistant_start=assistant_start)

    def render_system_message(self, content: str) -> str:
        return self._render_message(WINTERMUTE_SYSTEM, content)

    def render_user_message(self, content: str) -> str:
        return self._render_message(WINTERMUTE_USER, content)

    def render_assistant_message(
        self,
        *,
        final: str,
        task: str | None = None,
        verbosity: str | None = None,
        format: str | None = None,
        scratchpad: str | None = None,
    ) -> str:
        return (
            WINTERMUTE_ASSISTANT
            + self.render_completion(
                final=final,
                task=task,
                verbosity=verbosity,
                format=format,
                scratchpad=scratchpad,
            )
        )

    def render_completion(
        self,
        *,
        final: str,
        task: str | None = None,
        verbosity: str | None = None,
        format: str | None = None,
        scratchpad: str | None = None,
        include_terminator: bool = True,
    ) -> str:
        routing_prefix = self._render_routing_prefix(
            task=task,
            verbosity=verbosity,
            format=format,
        )
        if not scratchpad:
            continuation = f"{WINTERMUTE_FINAL}{final}" if self.uses_final_marker() else final
        elif self.uses_final_marker():
            continuation = f"{WINTERMUTE_SCRATCHPAD}{scratchpad}{WINTERMUTE_FINAL}{final}"
        else:
            continuation = f"{THINK_OPEN_TAG}{scratchpad}{THINK_CLOSE_TAG}{final}"
        content = routing_prefix + continuation
        return content + WINTERMUTE_EOT if include_terminator else content

    def parse_completion(
        self,
        text: str,
        assistant_prompt_format: AssistantPromptFormat | None = None,
    ) -> AssistantCompletion:
        if assistant_prompt_format is not None and self.uses_final_marker():
            text = self._render_assistant_prefix(assistant_prompt_format) + text
        cleaned = text.split(WINTERMUTE_EOT, maxsplit=1)[0].strip()
        if not self.uses_final_marker():
            return AssistantCompletion(final=cleaned)

        remainder = cleaned
        task: str | None = None
        verbosity: str | None = None
        format: str | None = None
        scratchpad: str | None = None

        if remainder.startswith(WINTERMUTE_TASK):
            remainder = remainder[len(WINTERMUTE_TASK):]
            boundaries = [
                index
                for marker in (
                    WINTERMUTE_VERBOSITY,
                    WINTERMUTE_FORMAT,
                    WINTERMUTE_SCRATCHPAD,
                    WINTERMUTE_FINAL,
                )
                if (index := remainder.find(marker)) >= 0
            ]
            if not boundaries:
                return AssistantCompletion(final=cleaned)
            boundary = min(boundaries)
            task = remainder[:boundary] or None
            remainder = remainder[boundary:]

        if remainder.startswith(WINTERMUTE_VERBOSITY):
            remainder = remainder[len(WINTERMUTE_VERBOSITY):]
            boundaries = [
                index
                for marker in (WINTERMUTE_FORMAT, WINTERMUTE_SCRATCHPAD, WINTERMUTE_FINAL)
                if (index := remainder.find(marker)) >= 0
            ]
            if not boundaries:
                return AssistantCompletion(final=cleaned)
            boundary = min(boundaries)
            verbosity = remainder[:boundary] or None
            remainder = remainder[boundary:]

        if remainder.startswith(WINTERMUTE_FORMAT):
            remainder = remainder[len(WINTERMUTE_FORMAT):]
            boundaries = [
                index
                for marker in (WINTERMUTE_SCRATCHPAD, WINTERMUTE_FINAL)
                if (index := remainder.find(marker)) >= 0
            ]
            if not boundaries:
                return AssistantCompletion(final=cleaned)
            boundary = min(boundaries)
            format = remainder[:boundary] or None
            remainder = remainder[boundary:]

        if remainder.startswith(WINTERMUTE_SCRATCHPAD):
            remainder = remainder[len(WINTERMUTE_SCRATCHPAD):]
            final_index = remainder.find(WINTERMUTE_FINAL)
            if final_index < 0:
                return AssistantCompletion(final=cleaned)
            scratchpad = remainder[:final_index] or None
            remainder = remainder[final_index:]

        if remainder.startswith(WINTERMUTE_FINAL):
            return AssistantCompletion(
                final=remainder[len(WINTERMUTE_FINAL):],
                task=task,
                verbosity=verbosity,
                format=format,
                scratchpad=scratchpad,
            )

        return AssistantCompletion(final=cleaned)

    def uses_final_marker(self) -> bool:
        return self.assistant_format == ASSISTANT_FORMAT_NORMALIZED

    def _render_assistant_prefix(
        self,
        assistant_prompt_format: AssistantPromptFormat,
    ) -> str:
        parts = [
            self._render_routing_prefix(
                task=assistant_prompt_format.forced_task,
                verbosity=assistant_prompt_format.forced_verbosity,
                format=assistant_prompt_format.forced_format,
            )
        ]

        last_token = assistant_prompt_format.last_prompt_control_token
        if (
            last_token is AssistantControlToken.TASK
            and assistant_prompt_format.forced_task is None
        ):
            parts.append(WINTERMUTE_TASK)
        elif (
            last_token is AssistantControlToken.VERBOSITY
            and assistant_prompt_format.forced_verbosity is None
        ):
            parts.append(WINTERMUTE_VERBOSITY)
        elif (
            last_token is AssistantControlToken.FORMAT
            and assistant_prompt_format.forced_format is None
        ):
            parts.append(WINTERMUTE_FORMAT)
        elif last_token is AssistantControlToken.SCRATCHPAD:
            parts.append(WINTERMUTE_SCRATCHPAD)
        elif last_token is AssistantControlToken.FINAL:
            parts.append(WINTERMUTE_FINAL)

        return "".join(parts)

    @staticmethod
    def _render_routing_prefix(
        *,
        task: str | None,
        verbosity: str | None,
        format: str | None,
    ) -> str:
        parts: list[str] = []
        if task:
            parts.extend((WINTERMUTE_TASK, task))
        if verbosity:
            parts.extend((WINTERMUTE_VERBOSITY, verbosity))
        if format:
            parts.extend((WINTERMUTE_FORMAT, format))
        return "".join(parts)

    def _render_context(
        self,
        context: ConversationContext,
        *,
        assistant_start: str,
    ) -> RenderedContextBlocks:
        return RenderedContextBlocks(
            system=self.render_system_message(context.system) if context.system else None,
            history=tuple(
                self.render_user_message(turn.user)
                + self.render_assistant_message(
                    final=turn.assistant,
                )
                for turn in context.history
            ),
            current=self.render_user_message(context.current_user) + assistant_start,
        )

    def _render_message(self, role: str, content: str) -> str:
        return f"{role}{content}{WINTERMUTE_EOT}"


@dataclass(frozen=True)
class PlainTextChatFormat:
    """Raw-completion fallback for tokenizers without a chat template."""

    def render_prompt(
        self,
        context: ConversationContext,
        assistant_prompt_format: AssistantPromptFormat | None = None,
    ) -> RenderedContextBlocks:
        del assistant_prompt_format
        return self._render_context(context)

    def render_completion(
        self,
        *,
        final: str,
        task: str | None = None,
        verbosity: str | None = None,
        format: str | None = None,
        scratchpad: str | None = None,
        include_terminator: bool = True,
    ) -> str:
        del include_terminator
        if not scratchpad:
            return final
        return f"{THINK_OPEN_TAG}{scratchpad}{THINK_CLOSE_TAG}{final}"

    def parse_completion(
        self,
        text: str,
        assistant_prompt_format: AssistantPromptFormat | None = None,
    ) -> AssistantCompletion:
        del assistant_prompt_format
        return AssistantCompletion(final=text.strip())

    def _render_context(self, context: ConversationContext) -> RenderedContextBlocks:
        return RenderedContextBlocks(
            system=f"{context.system}\n\n" if context.system else None,
            history=tuple(
                f"{turn.user}\n"
                f"{self.render_completion(final=turn.assistant)}\n\n"
                for turn in context.history
            ),
            current=context.current_user,
        )


class HuggingFaceChatFormat:
    """Chat format rendered by a Hugging Face tokenizer's official template."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        self._completion_terminator = self._resolve_completion_terminator()

    def render_prompt(
        self,
        context: ConversationContext,
        assistant_prompt_format: AssistantPromptFormat | None = None,
    ) -> RenderedContextBlocks:
        del assistant_prompt_format
        return self._render_context(context)

    def render_completion(
        self,
        *,
        final: str,
        task: str | None = None,
        verbosity: str | None = None,
        format: str | None = None,
        scratchpad: str | None = None,
        include_terminator: bool = True,
    ) -> str:
        if not scratchpad:
            content = final
        else:
            content = f"{THINK_OPEN_TAG}{scratchpad}{THINK_CLOSE_TAG}{final}"
        return content + self._completion_terminator if include_terminator else content

    def parse_completion(
        self,
        text: str,
        assistant_prompt_format: AssistantPromptFormat | None = None,
    ) -> AssistantCompletion:
        del assistant_prompt_format
        if self._completion_terminator:
            text = text.split(self._completion_terminator, maxsplit=1)[0]
        return AssistantCompletion(final=text.strip())

    def _render_context(self, context: ConversationContext) -> RenderedContextBlocks:
        messages: list[dict[str, str]] = []
        previous = ""
        system_block: str | None = None

        if context.system:
            messages.append({"role": "system", "content": context.system})
            previous = self._apply_template(messages, add_generation_prompt=False)
            system_block = previous

        history_blocks: list[str] = []
        for turn in context.history:
            messages.extend((
                {"role": "user", "content": turn.user},
                {
                    "role": "assistant",
                    "content": self.render_completion(
                        final=turn.assistant,
                        include_terminator=False,
                    ),
                },
            ))
            rendered = self._apply_template(messages, add_generation_prompt=False)
            history_blocks.append(rendered[len(previous):])
            previous = rendered

        messages.append({"role": "user", "content": context.current_user})
        rendered = self._apply_template(messages, add_generation_prompt=True)
        return RenderedContextBlocks(
            system=system_block,
            history=tuple(history_blocks),
            current=rendered[len(previous):],
        )

    def _resolve_completion_terminator(self) -> str:
        marker = "azerty"
        messages = [{"role": "user", "content": marker}]
        prefix = self._apply_template(messages, add_generation_prompt=True)
        messages.append({"role": "assistant", "content": marker})
        completed = self._apply_template(messages, add_generation_prompt=False)
        return completed[len(prefix) + len(marker):]

    def _apply_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> str:
        return cast(str, self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        ))


def build_hugging_face_chat_format(
    tokenizer: PreTrainedTokenizerBase,
) -> ChatFormat:
    if getattr(tokenizer, "chat_template", None):
        return HuggingFaceChatFormat(tokenizer)
    return PlainTextChatFormat()
