from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import Any, Callable, Optional

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from wintermute.ml.training.controller import parse_trainer_command
from wintermute.tools.logging import console_log
from .session import SlackChannelRef, SlackThreadSession
from .text import strip_slack_client_signature, truncate_for_slack


_SOCKET_RECEIVE_BUFFER_SIZE = 64 * 1024


class _SlackSdkLogHandler(logging.Handler):
    def __init__(self, *, is_closing: Callable[[], bool]) -> None:
        super().__init__(level=logging.WARNING)
        self._is_closing = is_closing

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage().strip()
        if not message:
            return

        exc = record.exc_info[1] if record.exc_info is not None else None
        if message.startswith("Failed to process a message:") and isinstance(exc, json.JSONDecodeError):
            suffix = " during shutdown" if self._is_closing() else ""
            console_log("slack runtime", f"ignored malformed socket payload{suffix}: {exc}")
            return

        console_log("slack sdk", f"[{record.levelname.lower()}] {message}")


def _build_slack_sdk_logger(*, is_closing: Callable[[], bool], logger_id: int) -> logging.Logger:
    logger = logging.Logger(f"textdata.slack.socket.{logger_id}", level=logging.WARNING)
    logger.propagate = False
    logger.addHandler(_SlackSdkLogHandler(is_closing=is_closing))
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


class SlackCommandInbox:
    def __init__(self) -> None:
        self._queue: Queue[str] = Queue()
        self._seen_ts: set[str] = set()

    def push(self, *, ts: str, text: str) -> bool:
        message_ts = ts.strip()
        if not message_ts or message_ts in self._seen_ts:
            return False

        command_text = text.strip()
        if not command_text:
            return False

        self._seen_ts.add(message_ts)
        self._queue.put(command_text)
        return True

    def drain(self) -> list[str]:
        commands: list[str] = []
        while True:
            try:
                commands.append(self._queue.get_nowait())
            except Empty:
                return commands


@dataclass(frozen=True)
class SlackThreadMessage:
    ts: str
    thread_ts: str
    user: str
    text: str
    is_thread_reply: bool


@dataclass(frozen=True)
class SlackPostedThreadReply:
    ts: str
    text: str


class SlackThreadMessageInbox:
    def __init__(self) -> None:
        self._queue: Queue[SlackThreadMessage] = Queue()
        self._seen_ts: set[str] = set()

    def push(
        self,
        *,
        ts: str,
        thread_ts: str,
        user: str,
        text: str,
        is_thread_reply: bool,
    ) -> bool:
        message_ts = ts.strip()
        if not message_ts or message_ts in self._seen_ts:
            return False

        clean_thread_ts = thread_ts.strip()
        clean_user = user.strip()
        clean_text = text.strip()
        if not clean_thread_ts or not clean_user or not clean_text:
            return False

        self._seen_ts.add(message_ts)
        self._queue.put(
            SlackThreadMessage(
                ts=message_ts,
                thread_ts=clean_thread_ts,
                user=clean_user,
                text=clean_text,
                is_thread_reply=bool(is_thread_reply),
            )
        )
        return True

    def drain(self) -> list[SlackThreadMessage]:
        messages: list[SlackThreadMessage] = []
        while True:
            try:
                messages.append(self._queue.get_nowait())
            except Empty:
                return messages


class SlackBoltRuntime:
    def __init__(
        self,
        *,
        bot_token: str,
        app_token: str,
        run_root: Path,
        channel: str,
        root_text: str,
    ) -> None:
        clean_bot_token = bot_token.strip()
        clean_app_token = app_token.strip()
        if not clean_bot_token:
            raise ValueError("Slack bot token must not be empty")
        if not clean_app_token:
            raise ValueError("Slack app token must not be empty")

        self._app_token = clean_app_token
        self._app = App(token=clean_bot_token, token_verification_enabled=False)
        self.session = SlackThreadSession(
            self._app.client,
            run_root=run_root,
            channel=channel,
            root_text=root_text,
        )
        self.commands = SlackCommandInbox()
        self._socket_handler: Optional[SocketModeHandler] = None
        self._lock = Lock()
        self._closing = False
        self._sdk_logger = _build_slack_sdk_logger(is_closing=lambda: self._closing, logger_id=id(self))
        self._register_thread_listener()

    def start(self) -> SlackBoltRuntime:
        self._closing = False
        self.session.ensure_thread()
        console_log(
            "slack runtime",
            f"listening on channel={self.session.channel} thread_ts={self.session.thread_ts}",
        )

        with self._lock:
            if self._socket_handler is not None:
                return self

            handler = SocketModeHandler(
                self._app,
                app_token=self._app_token,
                logger=self._sdk_logger,
                receive_buffer_size=_SOCKET_RECEIVE_BUFFER_SIZE,
            )
            handler.connect()
            self._socket_handler = handler

        return self

    def close(self) -> None:
        self._closing = True
        with self._lock:
            handler = self._socket_handler
            self._socket_handler = None

        if handler is not None:
            handler.close()
        _close_logger(self._sdk_logger)

    def _register_thread_listener(self) -> None:
        @self._app.event("message")
        def handle_thread_reply(event: dict[str, Any]) -> None:
            raw_text = str(event.get("text", ""))
            text = strip_slack_client_signature(raw_text)
            if self._should_trace_event(event, text):
                console_log("slack runtime", f"message event {self._describe_event(event, text)}")

            if event.get("subtype") is not None:
                if self._should_trace_event(event, text):
                    console_log("slack runtime", f"ignored message: subtype={event.get('subtype')}")
                return
            if not self.session.is_thread_reply(event):
                if self._should_trace_event(event, text):
                    console_log(
                        "slack runtime",
                        f"ignored message: expected channel={self.session.channel} thread_ts={self.session.thread_ts}",
                    )
                return

            if not self.commands.push(
                ts=str(event.get("ts", "")),
                text=text,
            ):
                if self._should_trace_event(event, text):
                    console_log("slack runtime", "ignored message: duplicate_or_empty")
                return

            console_log("slack runtime", f"queued command text=[{text.strip()}]")
            if parse_trainer_command(text) is not None:
                self._ack_command(text)

    def _ack_command(self, text: str) -> None:
        label = text.strip().split(maxsplit=1)[0].lower()
        try:
            self.session.post_reply(f"[info] [slack] command received: `{label}`")
        except Exception as exc:
            console_log("slack runtime", f"ack failed: {exc}")

    def _should_trace_event(self, event: dict[str, Any], text: str) -> bool:
        channel = str(event.get("channel", ""))
        thread_ts = str(event.get("thread_ts", ""))
        current_thread_ts = self.session.thread_ts or ""
        command = text.strip().split(maxsplit=1)[0].lower() if text.strip() else ""
        return (
            channel == self.session.channel
            or thread_ts == current_thread_ts
            or command in {"pause", "resume", "stop", "log", "metrics", "checkpoint", "params", "update_params"}
        )

    def _describe_event(self, event: dict[str, Any], text: str) -> str:
        compact = " ".join(text.strip().split())
        if len(compact) > 120:
            compact = compact[:117] + "..."
        return (
            f"channel={event.get('channel', '')} "
            f"thread_ts={event.get('thread_ts', '')} "
            f"ts={event.get('ts', '')} "
            f"subtype={event.get('subtype', '')} "
            f"text=[{compact}]"
        )


class SlackInferenceRuntime:
    _REPLY_MAX_CHARS = 20000
    _REPLY_MAX_LINES = 400

    def __init__(
        self,
        *,
        bot_token: str,
        app_token: str,
        channel: str,
    ) -> None:
        clean_bot_token = bot_token.strip()
        clean_app_token = app_token.strip()
        if not clean_bot_token:
            raise ValueError("Slack bot token must not be empty")
        if not clean_app_token:
            raise ValueError("Slack app token must not be empty")

        self._app_token = clean_app_token
        self._app = App(token=clean_bot_token, token_verification_enabled=False)
        self._channel_ref = SlackChannelRef(self._app.client, channel=channel)
        self.messages = SlackThreadMessageInbox()
        self._socket_handler: Optional[SocketModeHandler] = None
        self._lock = Lock()
        self._closing = False
        self._sdk_logger = _build_slack_sdk_logger(is_closing=lambda: self._closing, logger_id=id(self))
        self._register_channel_listener()

    @property
    def channel(self) -> str:
        return self._channel_ref.channel

    def start(self) -> SlackInferenceRuntime:
        self._closing = False
        channel = self._channel_ref.resolve()
        console_log("slack inference", f"listening on channel={channel}")

        with self._lock:
            if self._socket_handler is not None:
                return self

            handler = SocketModeHandler(
                self._app,
                app_token=self._app_token,
                logger=self._sdk_logger,
                receive_buffer_size=_SOCKET_RECEIVE_BUFFER_SIZE,
            )
            handler.connect()
            self._socket_handler = handler

        return self

    def close(self) -> None:
        self._closing = True
        with self._lock:
            handler = self._socket_handler
            self._socket_handler = None

        if handler is not None:
            handler.close()
        _close_logger(self._sdk_logger)

    def post_thread_reply(self, *, thread_ts: str, text: str) -> SlackPostedThreadReply:
        safe_text = truncate_for_slack(
            text.strip(),
            max_chars=self._REPLY_MAX_CHARS,
            max_lines=self._REPLY_MAX_LINES,
        )
        response = self._app.client.chat_postMessage(
            channel=self._channel_ref.resolve(),
            text=safe_text,
            thread_ts=thread_ts,
            mrkdwn=True,
        )
        return SlackPostedThreadReply(
            ts=str(response.get("ts", "")).strip(),
            text=safe_text,
        )

    def _register_channel_listener(self) -> None:
        @self._app.event("message")
        def handle_channel_message(event: dict[str, Any]) -> None:
            raw_text = str(event.get("text", ""))
            text = strip_slack_client_signature(raw_text)
            if self._should_trace_event(event):
                console_log("slack inference", f"message event {self._describe_event(event, text)}")

            if event.get("subtype") is not None:
                if self._should_trace_event(event):
                    console_log("slack inference", f"ignored message: subtype={event.get('subtype')}")
                return

            event_channel = str(event.get("channel", "")).strip()
            if event_channel != self.channel:
                return

            ts = str(event.get("ts", "")).strip()
            thread_ts = str(event.get("thread_ts", "")).strip() or ts
            user = str(event.get("user", "")).strip()
            is_thread_reply = bool(str(event.get("thread_ts", "")).strip() and thread_ts != ts)
            if not self.messages.push(
                ts=ts,
                thread_ts=thread_ts,
                user=user,
                text=text,
                is_thread_reply=is_thread_reply,
            ):
                if self._should_trace_event(event):
                    console_log("slack inference", "ignored message: duplicate_or_empty")
                return

            console_log(
                "slack inference",
                f"queued thread={thread_ts} user={user} text=[{self._compact_text(text)}]",
            )

    def _should_trace_event(self, event: dict[str, Any]) -> bool:
        return str(event.get("channel", "")).strip() == self.channel

    def _describe_event(self, event: dict[str, Any], text: str) -> str:
        return (
            f"channel={event.get('channel', '')} "
            f"thread_ts={event.get('thread_ts', '')} "
            f"ts={event.get('ts', '')} "
            f"subtype={event.get('subtype', '')} "
            f"user={event.get('user', '')} "
            f"text=[{self._compact_text(text)}]"
        )

    def _compact_text(self, text: str) -> str:
        compact = " ".join(text.strip().split())
        if len(compact) > 120:
            compact = compact[:117] + "..."
        return compact
