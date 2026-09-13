from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict

from wintermute.tools.params import get_int, get_optional_str, get_str
from wintermute.tools.secrets import resolve_secret_value


@dataclass
class SlackConfig:
    channel: str
    _bot_token: str = ""
    _app_token: str = ""
    push_interval_sec: float = 60.0
    metrics_history_macro_steps: int = -1
    secret_file: str | None = None
    bot_token_secret_id: str | None = None
    app_token_secret_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "SlackConfig":
        payload = dict(data)
        payload.pop("poll_interval_sec", None)

        channel = get_str(payload, "channel", "").strip()
        if not channel:
            raise ValueError("slack.channel is required")

        secret_file = _normalize_optional_str(get_optional_str(payload, "secret_file"))
        legacy_token = get_optional_str(payload, "token")
        explicit_bot_token = get_optional_str(payload, "bot_token")
        explicit_app_token = get_optional_str(payload, "app_token")
        bot_token_secret_id = _normalize_optional_str(get_optional_str(payload, "bot_token_secret_id"))
        app_token_secret_id = _normalize_optional_str(get_optional_str(payload, "app_token_secret_id"))
        metrics_history_macro_steps = get_int(payload, "metrics_history_macro_steps", -1)
        if metrics_history_macro_steps < -1 or metrics_history_macro_steps == 0:
            raise ValueError("slack.metrics_history_macro_steps must be -1 or a positive integer")

        if (bot_token_secret_id is not None or app_token_secret_id is not None) and secret_file is None:
            raise ValueError(
                "slack.secret_file is required when using bot_token_secret_id or app_token_secret_id"
            )

        config = cls(
            channel=channel,
            _bot_token=(explicit_bot_token or legacy_token or "").strip(),
            _app_token=(explicit_app_token or "").strip(),
            push_interval_sec=payload.get("push_interval_sec", 60.0),
            metrics_history_macro_steps=metrics_history_macro_steps,
            secret_file=secret_file,
            bot_token_secret_id=bot_token_secret_id,
            app_token_secret_id=app_token_secret_id,
        )

        _ = config.bot_token
        _ = config.app_token
        return config

    @cached_property
    def bot_token(self) -> str:
        return _resolve_token(
            explicit_value=self._bot_token,
            secret_id=self.bot_token_secret_id,
            secret_file=self.secret_file,
            field_name="slack.bot_token",
        )

    @cached_property
    def app_token(self) -> str:
        return _resolve_token(
            explicit_value=self._app_token,
            secret_id=self.app_token_secret_id,
            secret_file=self.secret_file,
            field_name="slack.app_token",
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "channel": self.channel,
            "push_interval_sec": self.push_interval_sec,
            "metrics_history_macro_steps": self.metrics_history_macro_steps,
        }
        if self.bot_token_secret_id is not None:
            data["bot_token_secret_id"] = self.bot_token_secret_id
        else:
            data["bot_token"] = self._bot_token

        if self.app_token_secret_id is not None:
            data["app_token_secret_id"] = self.app_token_secret_id
        else:
            data["app_token"] = self._app_token

        if self.secret_file is not None:
            data["secret_file"] = self.secret_file
        elif self.bot_token_secret_id is not None or self.app_token_secret_id is not None:
            raise ValueError(
                "slack.secret_file is required when using bot_token_secret_id or app_token_secret_id"
            )
        return data


def _normalize_optional_str(value: str | None) -> str | None:
    clean_value = (value or "").strip()
    return clean_value or None


def _resolve_token(
    *,
    explicit_value: str,
    secret_id: str | None,
    secret_file: str | None,
    field_name: str,
) -> str:
    clean_explicit_value = explicit_value.strip()
    if clean_explicit_value:
        return clean_explicit_value
    if secret_id is not None:
        if secret_file is None:
            raise ValueError(
                f"{field_name} uses secret id '{secret_id}' but slack.secret_file is not configured"
            )
        return resolve_secret_value(secret_id, path=secret_file).strip()
    raise ValueError(f"{field_name} is required")
