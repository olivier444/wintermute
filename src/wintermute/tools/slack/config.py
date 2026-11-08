from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict

from wintermute.tools.params import get_int, get_optional_str, get_str
from wintermute.tools.secrets import resolve_secret_value


@dataclass
class SlackConfig:
    channel: str
    secret_file: str
    bot_token_secret_id: str
    app_token_secret_id: str
    push_interval_sec: float = 60.0
    metrics_history_macro_steps: int = -1

    @classmethod
    def from_dict(cls, data: dict) -> "SlackConfig":
        channel = get_str(data, "channel", "").strip()
        if not channel:
            raise ValueError("slack.channel is required")

        secret_file = _normalize_optional_str(get_optional_str(data, "secret_file"))
        bot_token_secret_id = _normalize_optional_str(get_optional_str(data, "bot_token_secret_id"))
        app_token_secret_id = _normalize_optional_str(get_optional_str(data, "app_token_secret_id"))
        metrics_history_macro_steps = get_int(data, "metrics_history_macro_steps", -1)
        if metrics_history_macro_steps < -1 or metrics_history_macro_steps == 0:
            raise ValueError("slack.metrics_history_macro_steps must be -1 or a positive integer")

        if secret_file is None:
            raise ValueError("slack.secret_file is required")
        if bot_token_secret_id is None:
            raise ValueError("slack.bot_token_secret_id is required")
        if app_token_secret_id is None:
            raise ValueError("slack.app_token_secret_id is required")

        config = cls(
            channel=channel,
            push_interval_sec=data.get("push_interval_sec", 60.0),
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
        return resolve_secret_value(self.bot_token_secret_id, path=self.secret_file).strip()

    @cached_property
    def app_token(self) -> str:
        return resolve_secret_value(self.app_token_secret_id, path=self.secret_file).strip()

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "channel": self.channel,
            "push_interval_sec": self.push_interval_sec,
            "metrics_history_macro_steps": self.metrics_history_macro_steps,
        }
        data["bot_token_secret_id"] = self.bot_token_secret_id
        data["app_token_secret_id"] = self.app_token_secret_id
        data["secret_file"] = self.secret_file
        return data


def _normalize_optional_str(value: str | None) -> str | None:
    clean_value = (value or "").strip()
    return clean_value or None
