from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from wintermute.ml.inference.service import (
    InferenceConversationService,
    InferenceConversationState,
    InferenceServiceResult,
    InferenceTranscriptEvent,
)
from wintermute.ml.inference.session_log import InferenceSessionJsonStore
from wintermute.ml.inference.workspace import InferenceWorkspace
from wintermute.ml.tokenization.chat_format import AssistantCompletion
from wintermute.tools.logging import console_log
from wintermute.tools.slack.runtime import SlackInferenceRuntime, SlackPostedThreadReply, SlackThreadMessage
from wintermute.tools.slack.text import format_slack_code_block


class SlackInferenceEndpoint:
    def __init__(
        self,
        *,
        runtime: SlackInferenceRuntime,
        root: Path,
        models: Mapping[str, tuple[str, str | None]],
        default_model_name: str,
        device: str,
        system_prompt: str,
        poll_interval_sec: float,
        session_log_dir: str | Path,
    ) -> None:
        self._runtime = runtime
        self._root = Path(root)
        self._models = dict(models)
        self._active_model_name = default_model_name
        if self._active_model_name not in self._models:
            raise ValueError(f"default inference model '{self._active_model_name}' is not configured")
        self._device = device
        self._system_prompt = system_prompt
        self._use_chat_format = True
        self._force_final = False
        self._poll_interval_sec = float(poll_interval_sec)
        self._service = self._build_service(self._models[self._active_model_name])
        self._session_logs = InferenceSessionJsonStore(
            output_dir=session_log_dir,
            channel=runtime.channel,
        )

    def run_forever(self) -> None:
        console_log(
            "inference endpoint",
            f"online channel={self._runtime.channel} active_model={self._active_model_name}",
        )
        while True:
            messages = self._runtime.messages.drain()
            if not messages:
                time.sleep(self._poll_interval_sec)
                continue

            for message in messages:
                self._handle_message(message)

    def close(self) -> None:
        self._service.close()

    def _handle_message(self, message: SlackThreadMessage) -> None:
        try:
            if self._handle_endpoint_command(message):
                return
        except Exception as exc:
            console_log(
                "inference endpoint",
                f"endpoint command failed thread={message.thread_ts} user={message.user}: {exc}",
            )
            self._safe_post_reply(
                thread_ts=message.thread_ts,
                text=f"[info] {exc}",
            )
            return

        try:
            result = self._service.handle_message(
                conversation_id=message.thread_ts,
                text=message.text,
            )
        except Exception as exc:
            console_log(
                "inference endpoint",
                f"message handling failed thread={message.thread_ts} user={message.user}: {exc}",
            )
            self._safe_post_reply(
                thread_ts=message.thread_ts,
                text=f"[err] endpoint failed: {exc}",
            )
            return

        if result.created_session:
            origin = "existing_thread" if message.is_thread_reply else "new_thread"
            console_log(
                "inference endpoint",
                f"started session thread={message.thread_ts} origin={origin} model={self._active_model_name}",
            )

        self._append_non_assistant_events(message, result.transcript_events)

        posted_reply = self._safe_post_reply(
            thread_ts=message.thread_ts,
            text=_format_slack_inference_result(result),
        )
        if posted_reply is not None:
            self._append_assistant_event(
                thread_ts=message.thread_ts,
                posted_reply=posted_reply,
                transcript_events=result.transcript_events,
            )

    def _handle_endpoint_command(self, message: SlackThreadMessage) -> bool:
        clean_text = message.text.strip()
        chat_format_enabled = _parse_format_command(clean_text)
        if chat_format_enabled is not None:
            self._set_chat_format_enabled(chat_format_enabled)
            self._safe_post_reply(
                thread_ts=message.thread_ts,
                text=f"[info] chat_format={'on' if chat_format_enabled else 'off'} for the whole endpoint",
            )
            return True

        force_final = _parse_force_final_command(clean_text)
        if force_final is not None:
            self._set_force_final(force_final)
            self._safe_post_reply(
                thread_ts=message.thread_ts,
                text=f"[info] force_final={'on' if force_final else 'off'} for the whole endpoint",
            )
            return True

        device = _parse_device_command(clean_text)
        if device is not None:
            self._set_device(device)
            device_label = "gpu" if device == "cuda" else device
            self._safe_post_reply(
                thread_ts=message.thread_ts,
                text=f"[info] inference device is now '{device_label}' for the whole endpoint",
            )
            return True

        if clean_text == "!models":
            reply = self._describe_models()
            self._safe_post_reply(thread_ts=message.thread_ts, text=reply)
            return True

        if clean_text == "!again":
            self._switch_model_for_message(message, self._active_model_name)
            return True

        if clean_text == "!sp":
            self._safe_post_reply(
                thread_ts=message.thread_ts,
                text=self._describe_system_prompt(),
            )
            return True

        if clean_text == "!clearsp":
            self._clear_system_prompt()
            self._safe_post_reply(
                thread_ts=message.thread_ts,
                text="[info] system prompt cleared",
            )
            return True

        if clean_text.startswith("!sp "):
            self._update_system_prompt(clean_text[4:].strip())
            self._safe_post_reply(
                thread_ts=message.thread_ts,
                text="[info] system prompt updated",
            )
            return True

        if clean_text.startswith("!model "):
            self._switch_model_for_message(message, clean_text[7:].strip())
            return True

        if clean_text == "!model":
            self._safe_post_reply(
                thread_ts=message.thread_ts,
                text="[info] command 'model' requires a model name",
            )
            return True

        return False

    def _describe_models(self) -> str:
        lines: list[str] = []
        for model_name, target in self._models.items():
            marker = "*" if model_name == self._active_model_name else "-"
            run_id, external_model_name = target
            if external_model_name:
                lines.append(f"{marker} {model_name}: external={external_model_name} via task_run={run_id}")
            else:
                lines.append(f"{marker} {model_name}: run={run_id}")
        return "[info] available models\n" + format_slack_code_block("\n".join(lines))

    def _switch_model_for_message(self, message: SlackThreadMessage, requested_name: str) -> None:
        model_name, target = self._resolve_model(requested_name)

        preserved_states = self._service.export_conversation_states()
        regeneration_prompt = self._truncate_last_turn(
            preserved_states,
            conversation_id=message.thread_ts,
        )

        self._rebuild_service(
            target=target,
            preserved_states=preserved_states,
        )
        self._active_model_name = model_name

        console_log(
            "inference endpoint",
            f"switched model to {model_name}",
        )

        if regeneration_prompt is None:
            self._safe_post_reply(
                thread_ts=message.thread_ts,
                text=f"[info] active model is now '{model_name}'",
            )
            return

        result = self._service.handle_message(
            conversation_id=message.thread_ts,
            text=regeneration_prompt,
        )
        posted_reply = self._safe_post_reply(
            thread_ts=message.thread_ts,
            text=_format_slack_inference_result(result),
        )
        if posted_reply is not None:
            self._session_logs.replace_last_assistant_message(
                thread_ts=message.thread_ts,
                content=posted_reply.text,
                slack_ts=posted_reply.ts,
            )

    def _resolve_model(self, requested_name: str) -> tuple[str, tuple[str, str | None]]:
        clean_name = requested_name.strip().lower()
        if not clean_name:
            raise ValueError("model name must not be empty")

        for model_name, target in self._models.items():
            if model_name.lower() == clean_name:
                return model_name, target
        raise ValueError(f"unknown model '{requested_name.strip()}'")

    def _truncate_last_turn(
        self,
        states: dict[str, InferenceConversationState],
        *,
        conversation_id: str,
    ) -> str | None:
        state = states.get(conversation_id)
        if state is None or not state.session_state.turns:
            return None

        last_turn = state.session_state.turns[-1]
        states[conversation_id] = replace(
            state,
            session_state=replace(
                state.session_state,
                turns=state.session_state.turns[:-1],
            ),
        )
        return last_turn.user_text


    def _describe_system_prompt(self) -> str:
        clean_prompt = self._system_prompt.strip()
        if not clean_prompt:
            return "[info] system prompt is empty"
        return "[info] current system prompt\n" + format_slack_code_block(clean_prompt)

    def _update_system_prompt(self, new_system_prompt: str) -> None:
        clean_prompt = new_system_prompt.strip()
        if not clean_prompt:
            raise ValueError("system prompt must not be empty")

        self._set_system_prompt(clean_prompt)

        console_log(
            "inference endpoint",
            "updated system prompt",
        )

    def _clear_system_prompt(self) -> None:
        self._set_system_prompt("")

        console_log(
            "inference endpoint",
            "cleared system prompt",
        )

    def _set_system_prompt(self, new_system_prompt: str) -> None:
        preserved_states = self._service.export_conversation_states()
        preserved_states = {
            conversation_id: replace(state, system_prompt_logged=False)
            for conversation_id, state in preserved_states.items()
        }
        self._system_prompt = new_system_prompt.strip()
        self._rebuild_service(
            target=self._models[self._active_model_name],
            preserved_states=preserved_states,
        )

    def _set_chat_format_enabled(self, enabled: bool) -> None:
        next_value = bool(enabled)
        if self._use_chat_format == next_value:
            return

        preserved_states = self._service.export_conversation_states()
        self._use_chat_format = next_value
        self._rebuild_service(
            target=self._models[self._active_model_name],
            preserved_states=preserved_states,
        )

    def _set_force_final(self, enabled: bool) -> None:
        self._force_final = bool(enabled)
        self._service.set_force_final(self._force_final)

    def _set_device(self, device: str) -> None:
        if self._device == device:
            return

        preserved_states = self._service.export_conversation_states()
        previous_device = self._device
        self._device = device
        try:
            self._rebuild_service(
                target=self._models[self._active_model_name],
                preserved_states=preserved_states,
            )
        except Exception:
            self._device = previous_device
            raise

        console_log("inference endpoint", f"switched device to {device}")

    def _rebuild_service(
        self,
        *,
        target: tuple[str, str | None],
        preserved_states: dict[str, InferenceConversationState],
    ) -> None:
        next_service = self._build_service(target)
        try:
            next_service.restore_conversation_states(preserved_states)
        except Exception:
            next_service.close()
            raise

        previous_service = self._service
        self._service = next_service
        previous_service.close()

    def _build_service(self, target: tuple[str, str | None]) -> InferenceConversationService:
        run_id, external_model_name = target
        workspace = InferenceWorkspace.open(
            self._root,
            run_id,
            load_slack_config=True,
        )
        try:
            task_wrapper, model = workspace.create_components(
                device_override=self._device,
                external_model_name=external_model_name,
            )
        finally:
            workspace.close()

        return InferenceConversationService(
            task_wrapper=task_wrapper,
            model=model,
            system_prompt=self._system_prompt,
            use_chat_format=self._use_chat_format,
            force_final=self._force_final,
        )

    def _safe_post_reply(self, *, thread_ts: str, text: str) -> SlackPostedThreadReply | None:
        try:
            return self._runtime.post_thread_reply(thread_ts=thread_ts, text=text)
        except Exception as exc:
            console_log("inference endpoint", f"reply failed thread={thread_ts}: {exc}")
            return None

    def _append_non_assistant_events(
        self,
        message: SlackThreadMessage,
        events: tuple[InferenceTranscriptEvent, ...],
    ) -> None:
        for event in events:
            if event.role == "system":
                self._session_logs.append_system_message(
                    thread_ts=message.thread_ts,
                    content=event.content,
                )
            elif event.role == "user":
                self._session_logs.append_user_message(
                    thread_ts=message.thread_ts,
                    content=event.content,
                    slack_ts=message.ts,
                    slack_user=message.user,
                )

    def _append_assistant_event(
        self,
        *,
        thread_ts: str,
        posted_reply: SlackPostedThreadReply,
        transcript_events: tuple[InferenceTranscriptEvent, ...],
    ) -> None:
        if not any(event.role == "assistant" for event in transcript_events):
            return

        self._session_logs.append_assistant_message(
            thread_ts=thread_ts,
            content=posted_reply.text,
            slack_ts=posted_reply.ts,
        )


def _parse_format_command(text: str) -> bool | None:
    parts = text.strip().lower().split()
    if not parts or parts[0] != "!format":
        return None
    if len(parts) != 2 or parts[1] not in {"on", "off"}:
        raise ValueError("command 'format' requires 'on' or 'off'")
    return parts[1] == "on"


def _parse_force_final_command(text: str) -> bool | None:
    parts = text.strip().lower().split()
    if not parts or parts[0] != "!force_final":
        return None
    if len(parts) != 2 or parts[1] not in {"on", "off"}:
        raise ValueError("command 'force_final' requires 'on' or 'off'")
    return parts[1] == "on"


def _parse_device_command(text: str) -> str | None:
    parts = text.strip().lower().split()
    if not parts or parts[0] != "!device":
        return None
    if len(parts) != 2 or parts[1] not in {"cpu", "gpu"}:
        raise ValueError("command 'device' requires 'cpu' or 'gpu'")
    return "cuda" if parts[1] == "gpu" else "cpu"


def _format_slack_inference_result(result: InferenceServiceResult) -> str:
    if not isinstance(result.reply, AssistantCompletion):
        return result.reply_text
    completion = result.reply
    if (
        completion.task is None
        and completion.verbosity is None
        and completion.format is None
        and completion.scratchpad is None
    ):
        return completion.final

    sections: list[str] = []
    if completion.task is not None:
        sections.append(format_slack_code_block(f"Task: {completion.task}"))
    if completion.verbosity is not None:
        sections.append(format_slack_code_block(f"Verbosity: {completion.verbosity}"))
    if completion.format is not None:
        sections.append(format_slack_code_block(f"Format: {completion.format}"))
    if completion.scratchpad is not None:
        sections.append(format_slack_code_block(f"Scratchpad: {completion.scratchpad}"))
    sections.append(completion.final)
    return "\n\n".join(sections)
