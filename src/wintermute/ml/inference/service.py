from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping

import torch.nn as nn

from wintermute.ml.tokenization.chat_format import AssistantCompletion
from wintermute.ml.inference.commands import (
    InferenceCommand,
    InferenceCommandParseResult,
    inference_command_help_text,
    parse_inference_command,
)
from wintermute.tools.model import clear_cuda_memory, evaluating, unload_from_gpu
from wintermute.ml.tasks.factory import (
    InferenceSession,
    InferenceSessionState,
    InferenceSettings,
    TaskWrapper,
)


InferenceTranscriptRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class InferenceTranscriptEvent:
    role: InferenceTranscriptRole
    content: str


@dataclass(frozen=True)
class InferenceServiceResult:
    reply: str | AssistantCompletion
    transcript_events: tuple[InferenceTranscriptEvent, ...] = ()
    created_session: bool = False

    @property
    def reply_text(self) -> str:
        return self.reply.final if isinstance(self.reply, AssistantCompletion) else self.reply


@dataclass
class _ConversationState:
    session: InferenceSession
    system_prompt_logged: bool = False


@dataclass(frozen=True)
class InferenceConversationState:
    session_state: InferenceSessionState
    system_prompt_logged: bool = False


class InferenceConversationService:
    def __init__(
        self,
        *,
        task_wrapper: TaskWrapper,
        model: nn.Module,
        system_prompt: str | None = None,
        use_chat_format: bool = True,
        force_final: bool = False,
    ) -> None:
        self._task_wrapper = task_wrapper
        self._model = model
        self._system_prompt = system_prompt or ""
        self._use_chat_format = bool(use_chat_format)
        self._force_final = bool(force_final)
        self._sessions: dict[str, _ConversationState] = {}

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    def close(self) -> None:
        self._sessions.clear()
        try:
            self._model = unload_from_gpu(self._model)
        except Exception:
            clear_cuda_memory()

    def set_force_final(self, enabled: bool) -> None:
        self._force_final = bool(enabled)
        for state in self._sessions.values():
            state.session.settings.force_final = self._force_final

    def export_conversation_states(self) -> dict[str, InferenceConversationState]:
        return {
            conversation_id: InferenceConversationState(
                session_state=state.session.export_state(),
                system_prompt_logged=state.system_prompt_logged,
            )
            for conversation_id, state in self._sessions.items()
        }

    def restore_conversation_states(
        self,
        states: Mapping[str, InferenceConversationState],
    ) -> None:
        self._sessions.clear()
        for conversation_id, state in states.items():
            clean_conversation_id = conversation_id.strip()
            if not clean_conversation_id:
                continue
            settings = state.session_state.settings.clone()
            settings.force_final = self._force_final
            session = self._task_wrapper.start_inference_session(
                self._model,
                system_prompt=self._system_prompt,
                initial_state=replace(
                    state.session_state,
                    settings=settings,
                    use_chat_format=self._use_chat_format,
                ),
            )
            self._sessions[clean_conversation_id] = _ConversationState(
                session=session,
                system_prompt_logged=state.system_prompt_logged,
            )

    def handle_message(
        self,
        *,
        conversation_id: str,
        text: str,
    ) -> InferenceServiceResult:
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
                preview = state.session.infer(text)
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

        state, created_session = self._get_or_create_state(conversation_id)

        try:
            response = self._apply_control_command(state, command)
        except Exception as exc:
            response = f"[info] control command failed: {exc}"

        return InferenceServiceResult(
            reply=response,
            created_session=created_session,
        )

    def _apply_control_command(
        self,
        state: _ConversationState,
        command: InferenceCommand,
    ) -> str:
        session = state.session
        settings = session.settings

        if command.kind == "settings":
            return settings.describe(
                history_turn_count=session.history_turn_count,
                chat_format_enabled=self._use_chat_format,
            )
        if command.kind == "reset":
            session.reset()
            state.system_prompt_logged = False
            return "[info] conversation history reset for this thread"
        if command.kind == "max_tokens":
            assert isinstance(command.value, int)
            settings.max_new_tokens = command.value
            return f"[info] max_tokens={command.value}"
        if command.kind == "temperature":
            assert isinstance(command.value, float)
            settings.temperature = command.value
            return f"[info] temperature={command.value:.3f}"
        if command.kind == "top_k":
            assert isinstance(command.value, int) or command.value is None
            settings.top_k = command.value
            return f"[info] top_k={'off' if command.value is None else command.value}"
        if command.kind == "top_p":
            assert isinstance(command.value, float)
            settings.top_p = command.value
            return f"[info] top_p={command.value:.3f}"
        if command.kind == "greedy":
            assert isinstance(command.value, bool)
            settings.greedy = command.value
            return f"[info] greedy={'on' if command.value else 'off'}"
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
            initial_state=InferenceSessionState(
                settings=InferenceSettings(force_final=self._force_final),
                use_chat_format=self._use_chat_format,
            ),
        )
        state = _ConversationState(session=session, system_prompt_logged=False)
        self._sessions[conversation_id] = state
        return state, True

    def _take_system_prompt_event(
        self,
        state: _ConversationState,
    ) -> InferenceTranscriptEvent | None:
        if not self._use_chat_format:
            return None
        if state.system_prompt_logged:
            return None

        state.system_prompt_logged = True
        clean_system_prompt = self._system_prompt.strip()
        if not clean_system_prompt:
            return None

        return InferenceTranscriptEvent(role="system", content=clean_system_prompt)
