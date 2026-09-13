from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from slack_sdk.web import WebClient


class SlackChannelRef:
    def __init__(
        self,
        client: WebClient,
        *,
        channel: str,
    ) -> None:
        self._client = client
        self._requested_channel = channel.strip()
        self._channel = self._requested_channel
        if not self._requested_channel:
            raise ValueError("Slack channel must not be empty")

    @property
    def requested_channel(self) -> str:
        return self._requested_channel

    @property
    def channel(self) -> str:
        return self._channel

    def refs_match(self, *values: str) -> bool:
        aliases = [_channel_aliases(value) for value in values if value.strip()]
        if not aliases:
            return False
        requested_aliases = _channel_aliases(self._requested_channel)
        return any(requested_aliases & alias for alias in aliases)

    def resolve(self) -> str:
        if _looks_like_conversation_id(self._channel):
            return self._channel

        for candidate in (self._requested_channel, self._channel):
            resolved = _resolve_channel_id(self._client, candidate)
            if resolved:
                self._channel = resolved
                return self._channel

        return self._channel

    def set_resolved_channel(self, channel: Optional[str]) -> None:
        clean = (channel or "").strip()
        if clean:
            self._channel = clean


class SlackThreadSession:
    def __init__(
        self,
        client: WebClient,
        *,
        run_root: Path,
        channel: str,
        root_text: str,
    ) -> None:
        self._client = client
        self._run_root = Path(run_root)
        self._channel_ref = SlackChannelRef(client, channel=channel)
        self._root_text = root_text
        self._thread_ts: Optional[str] = None
        self._state_path = self._run_root / "slack.json"

    @property
    def channel(self) -> str:
        return self._channel_ref.channel

    @property
    def thread_ts(self) -> Optional[str]:
        if self._thread_ts is None:
            self._thread_ts = self._load_thread_ts()
        return self._thread_ts

    def ensure_thread(self) -> str:
        if self.thread_ts is not None:
            self._resolve_channel_reference()
            return self.thread_ts

        response = self._client.chat_postMessage(
            channel=self.channel,
            text=self._root_text,
            mrkdwn=True,
        )
        self._thread_ts = str(response["ts"])
        self._remember_channel(self._extract_channel_id(response.get("channel")))
        self._save_state(self._thread_ts)
        return self._thread_ts

    def update_root_message(self, text: str) -> None:
        thread_ts = self.ensure_thread()
        self._client.chat_update(
            channel=self._resolve_channel_reference(),
            ts=thread_ts,
            text=text,
            mrkdwn=True,
        )

    def post_reply(self, text: str) -> None:
        thread_ts = self.ensure_thread()
        self._client.chat_postMessage(
            channel=self._resolve_channel_reference(),
            text=text,
            thread_ts=thread_ts,
            mrkdwn=True,
        )

    def is_thread_reply(self, payload: dict[str, object]) -> bool:
        thread_ts = self.thread_ts
        if thread_ts is None:
            return False

        channel = str(payload.get("channel", ""))
        message_thread_ts = str(payload.get("thread_ts", ""))
        message_ts = str(payload.get("ts", ""))
        return channel == self.channel and message_thread_ts == thread_ts and message_ts not in {"", thread_ts}

    def _load_thread_ts(self) -> Optional[str]:
        if not self._state_path.exists():
            return None

        try:
            with self._state_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return None

        stored_channel = str(data.get("channel", "")).strip()
        requested_channel = str(data.get("requested_channel", "")).strip()
        if not self._channel_ref.refs_match(requested_channel):
            return None

        thread_ts = data.get("thread_ts")
        if thread_ts is None:
            return None
        if stored_channel:
            self._channel_ref.set_resolved_channel(stored_channel)
        return str(thread_ts)

    def _save_state(self, thread_ts: str) -> None:
        payload = {
            "requested_channel": self._channel_ref.requested_channel,
            "channel": self.channel,
            "thread_ts": thread_ts,
        }
        with self._state_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _resolve_channel_reference(self) -> str:
        resolved = self._channel_ref.resolve()
        self._remember_channel(resolved)
        return self.channel

    def _remember_channel(self, channel: Optional[str]) -> None:
        clean = (channel or "").strip()
        if not clean or clean == self.channel:
            return
        self._channel_ref.set_resolved_channel(clean)
        if self._thread_ts is not None:
            self._save_state(self._thread_ts)

    def _extract_channel_id(self, value: Any) -> Optional[str]:
        return _extract_channel_id(value)


def _channel_aliases(value: str) -> set[str]:
    clean = value.strip()
    if not clean:
        return set()
    aliases = {clean}
    aliases.add(clean.lstrip("#"))
    return aliases


def _resolve_channel_id(client: WebClient, value: str) -> Optional[str]:
    clean = value.strip()
    if not clean:
        return None
    if _looks_like_conversation_id(clean):
        return clean
    if _looks_like_user_id(clean):
        response = client.conversations_open(users=clean)
        return _extract_channel_id(response.get("channel"))
    return _resolve_named_channel(client, clean)


def _resolve_named_channel(client: WebClient, value: str) -> Optional[str]:
    channel_name = value.lstrip("#")
    if not channel_name:
        return None

    cursor: Optional[str] = None
    while True:
        response = client.conversations_list(
            types="public_channel,private_channel",
            exclude_archived=True,
            limit=1000,
            cursor=cursor,
        )
        for entry in response.get("channels", []):
            if not isinstance(entry, dict):
                continue
            if str(entry.get("name", "")).strip() != channel_name:
                continue
            return _extract_channel_id(entry.get("id"))

        metadata = response.get("response_metadata", {})
        if not isinstance(metadata, dict):
            return None
        cursor = str(metadata.get("next_cursor", "")).strip() or None
        if cursor is None:
            return None


def _extract_channel_id(value: Any) -> Optional[str]:
    if isinstance(value, str):
        clean = value.strip()
        return clean or None
    if isinstance(value, dict):
        nested = value.get("id")
        if isinstance(nested, str):
            clean = nested.strip()
            return clean or None
    return None


def _looks_like_conversation_id(value: str) -> bool:
    return value[:1] in {"C", "D", "G"} and len(value) > 1


def _looks_like_user_id(value: str) -> bool:
    return value[:1] in {"U", "W"} and len(value) > 1
