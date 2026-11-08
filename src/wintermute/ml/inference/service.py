from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping

import torch.nn as nn

from wintermute.ml.inference.conversation import (
    InferenceSession,
    InferenceSettings,
    InferenceSettingsDescription,
    InferenceTurn,
)
from wintermute.ml.tokenization.chat_format import AssistantCompletion
from wintermute.ml.inference.commands import (
    InferenceCommand,
    InferenceCommandParseResult,
    inference_command_help_text,
    parse_inference_command,
)
from wintermute.tools.model import clear_cuda_memory, evaluating, unload_from_gpu
from wintermute.ml.tasks.factory import TaskWrapper


InferenceTranscriptRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class InferenceTranscriptEvent:
    role: InferenceTranscriptRole
    content: str


@dataclass(frozen=True)
class InferenceServiceResult:
    reply: str | AssistantCompletion | InferenceSettingsDescription
    transcript_events: tuple[InferenceTranscriptEvent, ...] = ()
    created_session: bool = False

    @property
    def reply_text(self) -> str:
        if isinstance(self.reply, AssistantCompletion):
            return self.reply.final
        if isinstance(self.reply, InferenceSettingsDescription):
            return self.reply.as_text()
        return self.reply


@dataclass
class _ConversationState:
    session: InferenceSession
    system_prompt_logged: bool = False


@dataclass(frozen=True)
class InferenceConversationSnapshot:
    turns: tuple[InferenceTurn, ...]
    system_prompt_logged: bool = False


class InferenceConversationService:
    """Own a model and provide stateful inference sessions over it."""

    def __init__(
        self,
        *,
        task_wrapper: TaskWrapper,
        model: nn.Module,
        system_prompt: str | None = None,
        settings: InferenceSettings | None = None,
    ) -> None:
        self._task_wrapper = task_wrapper
        self._model = model
        self._system_prompt = system_prompt or ""
        self._settings = settings if settings is not None else InferenceSettings()
        self._settings.validate()
        self._sessions: dict[str, _ConversationState] = {}
        self._closed = False

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._sessions.clear()
        model = self._model
        try:
            unload_from_gpu(model)
        except Exception:
            pass
        finally:
            del self._model
            del model
            clear_cuda_memory()

    def __enter__(self) -> "InferenceConversationService":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def export_settings(self) -> InferenceSettings:
        self._ensure_open()
        return self._settings

    def export_conversation_snapshots(self) -> dict[str, InferenceConversationSnapshot]:
        self._ensure_open()
        return {
            conversation_id: InferenceConversationSnapshot(
                turns=state.session.export_turns(),
                system_prompt_logged=state.system_prompt_logged,
            )
            for conversation_id, state in self._sessions.items()
        }

    def restore_conversation_snapshots(
        self,
        snapshots: Mapping[str, InferenceConversationSnapshot],
    ) -> None:
        self._ensure_open()
        self._sessions.clear()
        for conversation_id, snapshot in snapshots.items():
            clean_conversation_id = conversation_id.strip()
            if not clean_conversation_id:
                continue
            session = self._task_wrapper.start_inference_session(
                self._model,
                system_prompt=self._system_prompt,
                initial_turns=snapshot.turns,
            )
            self._sessions[clean_conversation_id] = _ConversationState(
                session=session,
                system_prompt_logged=snapshot.system_prompt_logged,
            )

    def handle_message(
        self,
        *,
        conversation_id: str,
        text: str,
    ) -> InferenceServiceResult:
        self._ensure_open()
        clean_conversation_id = conversation_id.strip()
        if not clean_conversation_id:
            raise ValueError("conversation_id must not be empty")

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("message text must not be empty")

        parsed = parse_inference_command(clean_text)
        if parsed.is_control_message:
            return self._handle_control_message(clean_conversation_id, parsed)

        return self._handle_user_message(clean_conversation_id, clean_text)

    def regenerate_last(
        self,
        *,
        conversation_id: str,
    ) -> InferenceServiceResult | None:
        self._ensure_open()
        clean_conversation_id = conversation_id.strip()
        if not clean_conversation_id:
            raise ValueError("conversation_id must not be empty")

        state = self._sessions.get(clean_conversation_id)
        if state is None:
            return None

        turns = state.session.export_turns()
        if not turns:
            return None

        last_turn = turns[-1]
        state.session = self._task_wrapper.start_inference_session(
            self._model,
            system_prompt=self._system_prompt,
            initial_turns=turns[:-1],
        )
        return self._handle_user_message(clean_conversation_id, last_turn.user_text)

    def _handle_user_message(
        self,
        conversation_id: str,
        text: str,
    ) -> InferenceServiceResult:
        state, created_session = self._get_or_create_state(conversation_id)
        events: list[InferenceTranscriptEvent] = []

        system_event = self._take_system_prompt_event(state)
        if system_event is not None:
            events.append(system_event)

        events.append(InferenceTranscriptEvent(role="user", content=text))

        try:
            with evaluating(self._model):
                preview = state.session.infer(text, self._settings)
                reply: str | AssistantCompletion = (
                    preview.payload
                    if isinstance(preview.payload, AssistantCompletion)
                    else preview.as_text()
                )
        except Exception as exc:
            reply = f"[err] inference failed: {exc}"

        reply_text = reply.final if isinstance(reply, AssistantCompletion) else reply
        events.append(InferenceTranscriptEvent(role="assistant", content=reply_text))
        return InferenceServiceResult(
            reply=reply,
            transcript_events=tuple(events),
            created_session=created_session,
        )

    def _handle_control_message(
        self,
        conversation_id: str,
        parsed: InferenceCommandParseResult,
    ) -> InferenceServiceResult:
        if parsed.error is not None:
            return InferenceServiceResult(
                reply=(
                    f"[info] invalid control command: {parsed.error}\n"
                    f"{inference_command_help_text()}"
                )
            )

        command = parsed.command
        assert command is not None
        if command.kind == "help":
            return InferenceServiceResult(reply=inference_command_help_text())

        state = self._sessions.get(conversation_id)
        if command.kind == "settings":
            history_turn_count = (
                state.session.history_turn_count if state is not None else 0
            )
            return InferenceServiceResult(
                reply=self._settings.describe(
                    history_turn_count=history_turn_count,
                )
            )
        if command.kind == "reset":
            if state is not None:
                state.session.reset()
                state.system_prompt_logged = False
            return InferenceServiceResult(
                reply="[info] conversation history reset for this thread"
            )

        try:
            response = self._apply_global_command(command)
        except Exception as exc:
            response = f"[info] control command failed: {exc}"

        return InferenceServiceResult(reply=response)

    def _apply_global_command(
        self,
        command: InferenceCommand,
    ) -> str:
        settings = self._settings

        if command.kind == "max_tokens":
            assert isinstance(command.value, int)
            self._settings = replace(settings, max_new_tokens=command.value)
            return f"[info] max_tokens={command.value}"
        if command.kind == "temperature":
            assert isinstance(command.value, float)
            self._settings = replace(settings, temperature=command.value)
            return f"[info] temperature={command.value:.3f}"
        if command.kind == "top_k":
            assert isinstance(command.value, int) or command.value is None
            self._settings = replace(settings, top_k=command.value)
            return f"[info] top_k={'off' if command.value is None else command.value}"
        if command.kind == "top_p":
            assert isinstance(command.value, float)
            self._settings = replace(settings, top_p=command.value)
            return f"[info] top_p={command.value:.3f}"
        if command.kind == "greedy":
            assert isinstance(command.value, bool)
            self._settings = replace(settings, greedy=command.value)
            return f"[info] greedy={'on' if command.value else 'off'}"
        if command.kind == "format":
            assert isinstance(command.value, bool)
            self._settings = replace(settings, use_chat_format=command.value)
            return f"[info] chat_format={'on' if command.value else 'off'}"
        if command.kind == "force_final":
            assert isinstance(command.value, bool)
            self._settings = replace(settings, force_final=command.value)
            return f"[info] force_final={'on' if command.value else 'off'}"
        raise ValueError(f"unsupported control command: {command.kind}")

    def _get_or_create_state(
        self,
        conversation_id: str,
    ) -> tuple[_ConversationState, bool]:
        state = self._sessions.get(conversation_id)
        if state is not None:
            return state, False

        session = self._task_wrapper.start_inference_session(
            self._model,
            system_prompt=self._system_prompt,
        )
        state = _ConversationState(session=session, system_prompt_logged=False)
        self._sessions[conversation_id] = state
        return state, True

    def _take_system_prompt_event(
        self,
        state: _ConversationState,
    ) -> InferenceTranscriptEvent | None:
        if not self._settings.use_chat_format:
            return None
        if state.system_prompt_logged:
            return None

        state.system_prompt_logged = True
        clean_system_prompt = self._system_prompt.strip()
        if not clean_system_prompt:
            return None

        return InferenceTranscriptEvent(role="system", content=clean_system_prompt)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("inference conversation service is closed")
