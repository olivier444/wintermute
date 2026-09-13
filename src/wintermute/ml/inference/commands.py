from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


InferenceCommandKind = Literal[
    "help",
    "settings",
    "reset",
    "max_tokens",
    "temperature",
    "top_k",
    "top_p",
    "greedy",
]

INFERENCE_COMMAND_HELP_TEXT = """[info] inference commands
*Conversation*
• `!settings` — display the current settings
• `!reset` — clear this thread's history
• `!max_tokens 128`, `!temperature 0.7`, `!top_k 50` or `!top_k off`
• `!top_p 0.9`, `!greedy on` or `!greedy off`

*Models and system prompt*
• `!models`, `!model <name>`, `!again`
• `!sp`, `!sp <prompt>`, `!clearsp`

*Endpoint*
• `!device cpu`, `!device gpu`
• `!format on`, `!format off`
• `!force_final on`, `!force_final off`
• `!help`"""


@dataclass(frozen=True)
class InferenceCommand:
    kind: InferenceCommandKind
    value: int | float | bool | None = None


@dataclass(frozen=True)
class InferenceCommandParseResult:
    command: InferenceCommand | None = None
    error: str | None = None

    @property
    def is_control_message(self) -> bool:
        return self.command is not None or self.error is not None


def parse_inference_command(text: str) -> InferenceCommandParseResult:
    raw = text.strip()
    if not raw.startswith("!"):
        return InferenceCommandParseResult()

    payload = raw[1:].strip()
    if not payload:
        return InferenceCommandParseResult(error="empty command")

    parts = payload.split(maxsplit=1)
    command = parts[0].strip().lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command == "help":
        return _expect_no_argument(command, argument)
    if command in {"settings", "status"}:
        return _expect_no_argument("settings", argument)
    if command in {"reset", "clear"}:
        return _expect_no_argument("reset", argument)
    if command == "max_tokens":
        return _parse_positive_int(command, argument)
    if command == "temperature":
        return _parse_positive_float("temperature", argument)
    if command == "top_p":
        return _parse_positive_float("top_p", argument)
    if command == "top_k":
        return _parse_top_k(argument)
    if command == "greedy":
        return _parse_bool(command, argument)
    return InferenceCommandParseResult(error=f"unknown command '{command}'")


def inference_command_help_text() -> str:
    return INFERENCE_COMMAND_HELP_TEXT


def _expect_no_argument(kind: InferenceCommandKind, argument: str) -> InferenceCommandParseResult:
    if argument:
        return InferenceCommandParseResult(error=f"command '{kind}' does not take an argument")
    return InferenceCommandParseResult(command=InferenceCommand(kind=kind))


def _parse_positive_int(kind: InferenceCommandKind, argument: str) -> InferenceCommandParseResult:
    if not argument:
        return InferenceCommandParseResult(error=f"command '{kind}' requires an integer argument")

    try:
        value = int(argument)
    except ValueError:
        return InferenceCommandParseResult(error=f"invalid integer for '{kind}': {argument}")

    if value <= 0:
        return InferenceCommandParseResult(error=f"'{kind}' must be > 0")

    return InferenceCommandParseResult(command=InferenceCommand(kind=kind, value=value))


def _parse_positive_float(kind: InferenceCommandKind, argument: str) -> InferenceCommandParseResult:
    if not argument:
        return InferenceCommandParseResult(error=f"command '{kind}' requires a numeric argument")

    try:
        value = float(argument)
    except ValueError:
        return InferenceCommandParseResult(error=f"invalid float for '{kind}': {argument}")

    if value <= 0:
        return InferenceCommandParseResult(error=f"'{kind}' must be > 0")
    if kind == "top_p" and value > 1:
        return InferenceCommandParseResult(error="'top_p' must be <= 1")

    return InferenceCommandParseResult(command=InferenceCommand(kind=kind, value=value))


def _parse_top_k(argument: str) -> InferenceCommandParseResult:
    if not argument:
        return InferenceCommandParseResult(error="command 'top_k' requires an integer or 'off'")

    lowered = argument.lower()
    if lowered in {"off", "none"}:
        return InferenceCommandParseResult(command=InferenceCommand(kind="top_k", value=None))

    return _parse_positive_int("top_k", argument)


def _parse_bool(kind: InferenceCommandKind, argument: str) -> InferenceCommandParseResult:
    if not argument:
        return InferenceCommandParseResult(error=f"command '{kind}' requires 'on' or 'off'")

    lowered = argument.lower()
    if lowered in {"on", "true", "1", "yes"}:
        return InferenceCommandParseResult(command=InferenceCommand(kind=kind, value=True))
    if lowered in {"off", "false", "0", "no"}:
        return InferenceCommandParseResult(command=InferenceCommand(kind=kind, value=False))

    return InferenceCommandParseResult(error=f"invalid boolean for '{kind}': {argument}")
