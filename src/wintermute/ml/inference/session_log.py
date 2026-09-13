from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import json
from pathlib import Path
from typing import Any, Mapping

from wintermute.tools.files import json_save
from wintermute.tools.misc import utc_now


def _dt_to_iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat(timespec="milliseconds")


def _slack_ts_to_iso_utc(value: str | None) -> str:
    clean = (value or "").strip()
    if not clean:
        return _dt_to_iso_utc(utc_now())

    try:
        seconds = Decimal(clean)
    except InvalidOperation:
        return _dt_to_iso_utc(utc_now())

    whole_seconds = int(seconds.to_integral_value(rounding=ROUND_DOWN))
    fractional_seconds = seconds - Decimal(whole_seconds)
    microseconds = int(
        (fractional_seconds * Decimal(1_000_000)).to_integral_value(rounding=ROUND_DOWN)
    )
    dt = datetime.fromtimestamp(whole_seconds, tz=timezone.utc).replace(microsecond=microseconds)
    return _dt_to_iso_utc(dt)


def _clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    clean = str(value).strip()
    return clean or None


def _sanitize_thread_ts(thread_ts: str) -> str:
    clean = thread_ts.strip()
    if not clean:
        return "unknown-thread"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in clean)


@dataclass(frozen=True)
class InferenceSessionMessageRecord:
    role: str
    content: str
    timestamp_utc: str
    slack_ts: str | None = None
    slack_user: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InferenceSessionMessageRecord | None":
        role = str(data.get("role", "")).strip()
        content = str(data.get("content", ""))
        timestamp_utc = str(data.get("timestamp_utc", "")).strip()
        if not role or not timestamp_utc:
            return None
        return cls(
            role=role,
            content=content,
            timestamp_utc=timestamp_utc,
            slack_ts=_clean_optional_str(data.get("slack_ts")),
            slack_user=_clean_optional_str(data.get("slack_user")),
        )


@dataclass
class InferenceThreadLog:
    channel: str
    thread_ts: str
    messages: list[InferenceSessionMessageRecord] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        default_channel: str,
        default_thread_ts: str,
    ) -> "InferenceThreadLog":
        messages: list[InferenceSessionMessageRecord] = []
        raw_messages = data.get("messages", [])
        if isinstance(raw_messages, list):
            for entry in raw_messages:
                if not isinstance(entry, dict):
                    continue
                message = InferenceSessionMessageRecord.from_dict(entry)
                if message is not None:
                    messages.append(message)

        return cls(
            channel=str(data.get("channel", default_channel)).strip() or default_channel,
            thread_ts=str(data.get("thread_ts", default_thread_ts)).strip() or default_thread_ts,
            messages=messages,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "thread_ts": self.thread_ts,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "timestamp_utc": message.timestamp_utc,
                    "slack_ts": message.slack_ts,
                    "slack_user": message.slack_user,
                }
                for message in self.messages
            ],
        }


class InferenceSessionJsonStore:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        channel: str,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._channel = channel
        self._threads: dict[str, InferenceThreadLog] = {}

    def append_system_message(self, *, thread_ts: str, content: str) -> None:
        clean_content = content.strip()
        if not clean_content:
            return

        thread_log = self._get_thread_log(thread_ts)
        thread_log.messages.append(
            InferenceSessionMessageRecord(
                role="system",
                content=clean_content,
                timestamp_utc=_dt_to_iso_utc(utc_now()),
            )
        )
        self._save_thread_log(thread_log)

    def append_user_message(
        self,
        *,
        thread_ts: str,
        content: str,
        slack_ts: str,
        slack_user: str,
    ) -> None:
        thread_log = self._get_thread_log(thread_ts)
        thread_log.messages.append(
            InferenceSessionMessageRecord(
                role="user",
                content=content,
                timestamp_utc=_slack_ts_to_iso_utc(slack_ts),
                slack_ts=slack_ts,
                slack_user=slack_user,
            )
        )
        self._save_thread_log(thread_log)

    def append_assistant_message(self, *, thread_ts: str, content: str, slack_ts: str | None) -> None:
        thread_log = self._get_thread_log(thread_ts)
        thread_log.messages.append(
            InferenceSessionMessageRecord(
                role="assistant",
                content=content,
                timestamp_utc=_slack_ts_to_iso_utc(slack_ts),
                slack_ts=_clean_optional_str(slack_ts),
            )
        )
        self._save_thread_log(thread_log)

    def replace_last_assistant_message(
        self,
        *,
        thread_ts: str,
        content: str,
        slack_ts: str | None,
    ) -> None:
        thread_log = self._get_thread_log(thread_ts)
        replacement = InferenceSessionMessageRecord(
            role="assistant",
            content=content,
            timestamp_utc=_slack_ts_to_iso_utc(slack_ts),
            slack_ts=_clean_optional_str(slack_ts),
        )
        for index in range(len(thread_log.messages) - 1, -1, -1):
            if thread_log.messages[index].role == "assistant":
                thread_log.messages[index] = replacement
                self._save_thread_log(thread_log)
                return

        thread_log.messages.append(replacement)
        self._save_thread_log(thread_log)

    def _get_thread_log(self, thread_ts: str) -> InferenceThreadLog:
        clean_thread_ts = thread_ts.strip()
        if not clean_thread_ts:
            raise ValueError("thread_ts must not be empty")

        cached = self._threads.get(clean_thread_ts)
        if cached is not None:
            return cached

        path = self._file_path(clean_thread_ts)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                payload = {}
        else:
            payload = {}

        if isinstance(payload, dict):
            thread_log = InferenceThreadLog.from_dict(
                payload,
                default_channel=self._channel,
                default_thread_ts=clean_thread_ts,
            )
        else:
            thread_log = InferenceThreadLog(
                channel=self._channel,
                thread_ts=clean_thread_ts,
            )

        self._threads[clean_thread_ts] = thread_log
        return thread_log

    def _save_thread_log(self, thread_log: InferenceThreadLog) -> None:
        json_save(
            thread_log,
            directory=self._output_dir.as_posix(),
            file_name=self._file_path(thread_log.thread_ts).name,
        )

    def _file_path(self, thread_ts: str) -> Path:
        file_name = f"{_sanitize_thread_ts(thread_ts)}.json"
        return self._output_dir / file_name
