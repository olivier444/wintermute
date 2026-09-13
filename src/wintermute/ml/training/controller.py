from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Literal, List

import json

from wintermute.tools.logging import console_log
from wintermute.tools.misc import FileController


CommandKind = Literal["help", "pause", "resume", "stop", "update_params", "log", "metrics", "checkpoint"]

SLACK_COMMAND_HELP_TEXT = """[info] *Training commands*
• `help` — display this message
• `pause` · `resume` · `stop` — control training
• `log` · `probes` — publish the latest model previews (`probes` is an alias of `log`)
• `metrics [steps_ago]` — publish metrics for now, or an earlier macro step
• `checkpoint [name]` — preserve the latest checkpoint under a name

*Update parameters* — send a JSON object with `params`:
`params {"max_lr": 0.0001, "grad_clip": 0.5}`

• *Learning rate:* `start_lr`, `end_lr`, `max_lr`, `warmup_length_ratio`, `cos_decay_start_ratio`
• *Optimizer:* `grad_clip`, `beta1`, `beta2`, `weight_decay`
• *Training:* `max_training_units`, `checkpoint_every_batch`, `grad_accum_size_units`
• *Gradient-noise diagnostics:* `gradient_noise_update_batch_count` (default `4`; `0` disables and resets it), `gradient_noise_modulus` (default `1`; estimate every N macro steps). Example: `params {"gradient_noise_update_batch_count": 4, "gradient_noise_modulus": 10}`
• *Evaluation cadence:* `evaluation_moduli`, e.g. `params {"evaluation_moduli": {"secondary_eval": 10}}`; the primary evaluation must stay at `1`.
• *Benchmark cadence:* `benchmark_every_n_checkpoints`, e.g. `params {"benchmark_every_n_checkpoints": 4}`; changing it resets the cadence.

Commands are case-insensitive. `params` also accepts `update_params`."""


@dataclass(frozen=True)
class TrainerCommand:
    kind: CommandKind
    params: Optional[dict[str, Any]] = None
    metrics_offset: int = 0
    checkpoint_name: Optional[str] = None


class TrainerController(Protocol):
    def poll(self) -> List[TrainerCommand]: ...


class CommandTextSource(Protocol):
    def drain(self) -> List[str]: ...


class FileTrainerController(TrainerController):
    def __init__(self, control_dir: Path, check_interval_sec: float = 10.0) -> None:
        self._controller = FileController(control_dir)
        self._check_interval_sec = float(check_interval_sec)
        self._pause_active = False

    def poll(self) -> List[TrainerCommand]:
        commands: List[TrainerCommand] = []

        param_req = self._controller.get_request(
            ".params",
            log_changes=False,
            check_interval_sec=self._check_interval_sec,
        )
        if param_req.Active and param_req.Params is not None:
            commands.append(TrainerCommand("update_params", dict(param_req.Params)))

        pause_active = self._controller.has_request(
            ".pause",
            clear=False,
            log_changes=False,
            check_interval_sec=self._check_interval_sec * 0.95,
        )
        if pause_active != self._pause_active:
            commands.append(TrainerCommand("pause" if pause_active else "resume"))
            self._pause_active = pause_active

        if self._controller.has_request(
            ".stop",
            log_changes=False,
            check_interval_sec=self._check_interval_sec,
        ):
            commands.append(TrainerCommand("stop"))

        return commands


class CompositeTrainerController(TrainerController):
    def __init__(self, controllers: List[TrainerController]) -> None:
        self._controllers = list(controllers)

    def poll(self) -> List[TrainerCommand]:
        commands: List[TrainerCommand] = []
        for controller in self._controllers:
            commands.extend(controller.poll())
        return commands


class SlackTrainerController(TrainerController):
    def __init__(
        self,
        command_source: CommandTextSource,
    ) -> None:
        self._command_source = command_source

    def poll(self) -> List[TrainerCommand]:
        commands: List[TrainerCommand] = []
        for text in self._command_source.drain():
            console_log("slack controller", f"command text detected: [{text}]")
            command = parse_trainer_command(text)
            if command is not None:
                commands.append(command)
        return commands


def parse_trainer_command(text: str) -> Optional[TrainerCommand]:
    raw = text.strip()
    if not raw:
        return None

    parts = raw.split(maxsplit=1)
    command = parts[0].strip().lower()

    if command == "help":
        return TrainerCommand("help") if len(parts) == 1 else None
    if command == "pause":
        return TrainerCommand("pause")
    if command == "resume":
        return TrainerCommand("resume")
    if command == "stop":
        return TrainerCommand("stop")
    if command in {"log", "probes"}:
        return TrainerCommand("log")
    if command == "metrics":
        if len(parts) == 1:
            return TrainerCommand("metrics")
        try:
            metrics_offset = int(parts[1])
        except ValueError:
            return None
        if metrics_offset < 0:
            return None
        return TrainerCommand("metrics", metrics_offset=metrics_offset)
    if command == "checkpoint":
        if len(parts) < 2 or not parts[1].strip():
            return None
        return TrainerCommand("checkpoint", checkpoint_name=parts[1].strip())
    if command not in {"params", "update_params"}:
        return None
    if len(parts) < 2:
        return None

    payload = parts[1].strip()
    if payload.startswith("```"):
        payload = _strip_code_fence(payload)

    try:
        params = json.loads(payload)
    except Exception:
        return None

    if not isinstance(params, dict):
        return None

    return TrainerCommand("update_params", params)


def slack_command_help_text() -> str:
    return SLACK_COMMAND_HELP_TEXT


def slack_command_help_hint() -> str:
    return "Reply `help` in this thread for training commands."


def _strip_code_fence(payload: str) -> str:
    lines = payload.strip().splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return payload
