from __future__ import annotations

from enum import Enum
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Dict, List
import json
import re
import time
from torch.utils.tensorboard import SummaryWriter
from wintermute.tools.logging import console_log
from wintermute.tools.misc import json_default
from wintermute.tools.slack.text import format_slack_code_block, truncate_for_slack
from wintermute.ml.runs.checkpoints import RunCheckpoints
from wintermute.ml.training.controller import slack_command_help_hint


class LogFormat(Enum):
    TEXT = "text"
    JSON = "json"


class LogLevel(Enum):
    DATA = "data"
    INFO = "info"
    WARN = "warn"
    ERR = "err"


class Logger(Protocol):
    def start_fit(self) -> None: ...

    def log_scalar(
        self,
        name: str,
        value: float,
        step: int,
        level: LogLevel = LogLevel.DATA,
    ) -> None: ...

    def log_text(
        self,
        name: str,
        text: str,
        step: Optional[int] = None,
        format: LogFormat = LogFormat.TEXT,
        level: LogLevel = LogLevel.INFO,
    ) -> None: ...

    def log_params_text(
        self,
        params: Mapping[str, Any],
        name: Optional[str] = None,
        level: LogLevel = LogLevel.DATA,
    ) -> None: ...

    def publish_preview_snapshot(self, previews: Mapping[str, List[str]]) -> None: ...

    def publish_metrics_snapshot(self, steps_ago: int = 0) -> None: ...

    def close(self) -> None: ...


class SlackMessagePublisher(Protocol):
    def update_root_message(self, text: str) -> None: ...
    def post_reply(self, text: str) -> None: ...


class TensorBoardRunLogger(Logger):
    def __init__(
        self,
        run_root: Path,
        *,
        primary_eval_suite_name: str,
        eval_suite_names: List[str],
        routing_label_eval_suite_names: Mapping[str, List[str]],
        benchmark_task_names: List[str],
    ):
        tb_dir = run_root / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(tb_dir))
        self._writer.add_custom_scalars(
            self._custom_scalars_layout(
                primary_eval_suite_name=primary_eval_suite_name,
                eval_suite_names=eval_suite_names,
                routing_label_eval_suite_names=routing_label_eval_suite_names,
                benchmark_task_names=benchmark_task_names,
            )
        )

        self._log_root = run_root / "log"
        self._log_root.mkdir(parents=True, exist_ok=True)
        
        self._metrics_path = self._log_root / "metrics.jsonl"

    @staticmethod
    def _custom_scalars_layout(
        *,
        primary_eval_suite_name: str,
        eval_suite_names: List[str],
        routing_label_eval_suite_names: Mapping[str, List[str]],
        benchmark_task_names: List[str],
    ) -> Dict[str, Dict[str, list[Any]]]:
        primary_eval_loss = rf"^loss/eval/{re.escape(primary_eval_suite_name)}$"
        routing_label_categories = {
            f"Evaluation {label_name}-label accuracy": TensorBoardRunLogger._eval_suite_charts(
                f"{label_name}_label_acc",
                suite_names,
            )
            for label_name, suite_names in routing_label_eval_suite_names.items()
            if suite_names
        }

        layout: Dict[str, Dict[str, list[Any]]] = {
            "Overview": {
                "Train and primary evaluation loss": [
                    "Multiline",
                    [r"^loss/train$", primary_eval_loss],
                ],
                "Learning rate": ["Multiline", [r"^optimizer/learning_rate$"]],
                "Throughput": ["Multiline", [r"^perf/units_per_sec$"]],
            },
            "Evaluation loss": TensorBoardRunLogger._eval_suite_charts(
                "loss",
                eval_suite_names,
            ),
            "Evaluation bits per byte": TensorBoardRunLogger._eval_suite_charts(
                "bpb",
                eval_suite_names,
            ),
            "Evaluation bits per byte by position": {
                suite_name: [
                    "Multiline",
                    [rf"^bpb_pos_.+/eval/{re.escape(suite_name)}$"],
                ]
                for suite_name in eval_suite_names
            },
            "Evaluation top-10 accuracy": TensorBoardRunLogger._eval_suite_charts(
                "top10_acc",
                eval_suite_names,
            ),
            **routing_label_categories,
            **(
                {
                    "Benchmarks": {
                        task_name: [
                            "Multiline",
                            [rf"^.+/benchmark/{re.escape(task_name)}(?:/.+)?$"],
                        ]
                        for task_name in benchmark_task_names
                    }
                }
                if benchmark_task_names
                else {}
            ),
            "Optimization": {
                "Training loss": [
                    "Multiline",
                    [r"^loss/train$", r"^loss/train_last_batch$"],
                ],
                "Update direction": [
                    "Multiline",
                    [
                        r"^optimizer/neg_grad_update_cos$",
                        r"^optimizer/grad_cos_prev$",
                        r"^optimizer/update_cos_prev$",
                    ],
                ],
                "Relative update scale": [
                    "Multiline",
                    [
                        r"^optimizer/delta_param_to_param$",
                        r"^optimizer/grad_to_param$",
                    ],
                ],
            },
            "Runtime": {
                "Current GPU memory": [
                    "Multiline",
                    [r"^gpu/mem_allocated_mb$", r"^gpu/mem_reserved_mb$"],
                ],
                "Peak GPU memory": [
                    "Multiline",
                    [
                        r"^gpu/mem_max_allocated_mb$",
                        r"^gpu/mem_max_reserved_mb$",
                    ],
                ],
                "Step timing": [
                    "Multiline",
                    [
                        r"^perf/sec_per_step$",
                        r"^perf/eval_checkpoint_sec$",
                        r"^perf/monitor_checkpoint_sec$",
                        r"^perf/benchmark_checkpoint_sec$",
                    ],
                ],
                "Gradient-noise batch readiness": [
                    "Multiline",
                    [
                        r"^grad_noise/update_batch_count_available$",
                        r"^grad_noise/update_batch_count_required$",
                    ],
                ],
            },
        }
        return layout

    @staticmethod
    def _eval_suite_charts(
        metric: str,
        suite_names: List[str],
    ) -> Dict[str, list[Any]]:
        return {
            suite_name: [
                "Multiline",
                [rf"^{re.escape(metric)}/eval/{re.escape(suite_name)}$"],
            ]
            for suite_name in suite_names
        }


    def log_scalar(
        self,
        name: str,
        value: float,
        step: int,
        level: LogLevel = LogLevel.DATA,
    ) -> None:
        step_i = int(step)
        val_f = float(value)
        self._writer.add_scalar(name, val_f, global_step=step_i)

        rec = {
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": level.value,
            "name": name,
            "value": val_f,
            "step": step_i,
        }
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def start_fit(self) -> None:
        return


    def log_text(
        self,
        name: str,
        text: str,
        step: Optional[int] = None,
        format: LogFormat = LogFormat.TEXT,
        level: LogLevel = LogLevel.INFO,
    ) -> None:
        if level is not LogLevel.DATA and name.startswith("probes/"):
            return

        gs = 0 if step is None else int(step)
        formatted_text = self._format(text, format)
        self._writer.add_text(self._tag_name(name), formatted_text, global_step=gs)

        text_dir = self._log_root / level.value / name
        text_dir.mkdir(parents=True, exist_ok=True)

        suffix = "" if step is None else f"_{step}"
        extension = ".json" if format is LogFormat.JSON else ".txt"
        text_file = text_dir / f"{text_dir.name}{suffix}{extension}"

        with text_file.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


    def _tag_name(self, name: str) -> str:
        return name if "/" in name else f"events/{name}"


    def _format(self, text: str, format: LogFormat) -> str:
        if format == LogFormat.JSON:
            return f"```json\n{text}\n```"
        
        return f"```text\n{text}\n```"


    def log_params_text(
        self,
        params: Mapping[str, Any],
        name: Optional[str] = None,
        level: LogLevel = LogLevel.DATA,
    ) -> None:
        dumped = json.dumps(dict(params), indent=2, ensure_ascii=False, default=json_default)
        self.log_text(name or "run/params", dumped, format=LogFormat.JSON, level=level)

    def publish_preview_snapshot(self, previews: Mapping[str, List[str]]) -> None:
        return


    def publish_metrics_snapshot(self, steps_ago: int = 0) -> None:
        return

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


class CompositeLogger(Logger):
    def __init__(self, loggers: List[Logger]) -> None:
        self._loggers = list(loggers)

    def log_scalar(
        self,
        name: str,
        value: float,
        step: int,
        level: LogLevel = LogLevel.DATA,
    ) -> None:
        for logger in self._loggers:
            self._safe_call(logger, "log_scalar", name, value, step, level=level)

    def start_fit(self) -> None:
        for logger in self._loggers:
            self._safe_call(logger, "start_fit")

    def log_text(
        self,
        name: str,
        text: str,
        step: Optional[int] = None,
        format: LogFormat = LogFormat.TEXT,
        level: LogLevel = LogLevel.INFO,
    ) -> None:
        for logger in self._loggers:
            self._safe_call(logger, "log_text", name, text, step=step, format=format, level=level)

    def log_params_text(
        self,
        params: Mapping[str, Any],
        name: Optional[str] = None,
        level: LogLevel = LogLevel.DATA,
    ) -> None:
        for logger in self._loggers:
            self._safe_call(logger, "log_params_text", params, name=name, level=level)

    def publish_preview_snapshot(self, previews: Mapping[str, List[str]]) -> None:
        for logger in self._loggers:
            self._safe_call(logger, "publish_preview_snapshot", previews)

    def publish_metrics_snapshot(self, steps_ago: int = 0) -> None:
        for logger in self._loggers:
            self._safe_call(logger, "publish_metrics_snapshot", steps_ago)

    def close(self) -> None:
        for logger in self._loggers:
            self._safe_call(logger, "close")

    def _safe_call(self, logger: Logger, method_name: str, *args: Any, **kwargs: Any) -> None:
        try:
            method = getattr(logger, method_name)
            method(*args, **kwargs)
        except Exception as exc:
            console_log("composite logger", f"{type(logger).__name__}.{method_name} failed: {exc}")


class SlackRunLogger(Logger):
    _STATUS_MAX_CHARS = 500
    _STATUS_MAX_LINES = 20
    _REPLY_MAX_CHARS = 2000
    _REPLY_MAX_LINES = 60
    _PROBES_MAX_CHARS = 35000
    _PROBES_MAX_LINES = 1200
    _METRICS_MAX_CHARS = 12000
    _METRICS_MAX_LINES = 400

    def __init__(
        self,
        session: SlackMessagePublisher,
        run_id: str,
        checkpoints: RunCheckpoints,
        notify_interval_sec: float = 120.0,
        metrics_history_macro_steps: int = -1,
    ) -> None:
        self._session = session
        self._run_id = run_id
        self._checkpoints = checkpoints
        self._notify_interval_sec = max(5.0, float(notify_interval_sec))
        self._latest_scalars: Dict[str, float] = {}
        self._latest_step = 0
        if metrics_history_macro_steps < -1 or metrics_history_macro_steps == 0:
            raise ValueError("metrics_history_macro_steps must be -1 or a positive integer")
        self._metrics_history_macro_steps = metrics_history_macro_steps
        self._metrics_history: Dict[int, Dict[str, float]] = {}
        self._latest_status: Optional[str] = None
        self._fit_started_at: Optional[float] = None
        self._last_summary_push = 0.0
        self._warned_at = 0.0
        self._push_summary(force=True)

    def start_fit(self) -> None:
        self._fit_started_at = time.monotonic()
        self._push_summary(force=True)

    def log_scalar(
        self,
        name: str,
        value: float,
        step: int,
        level: LogLevel = LogLevel.DATA,
    ) -> None:
        self._latest_scalars[name] = float(value)
        step_i = int(step)
        self._latest_step = max(self._latest_step, step_i)
        self._metrics_history[step_i] = dict(self._latest_scalars)
        self._trim_metrics_history()
        self._push_summary()

    def log_text(
        self,
        name: str,
        text: str,
        step: Optional[int] = None,
        format: LogFormat = LogFormat.TEXT,
        level: LogLevel = LogLevel.INFO,
    ) -> None:
        if level == LogLevel.DATA:
            return
                
        self._latest_status = self._format_slack_text(
            name,
            text,
            format,
            level,
            max_chars=self._STATUS_MAX_CHARS,
            max_lines=self._STATUS_MAX_LINES,
        )
        self._push_summary(force=True)
        self._safe_post_reply(
            self._format_slack_text(
                name,
                text,
                format,
                level,
                max_chars=self._REPLY_MAX_CHARS,
                max_lines=self._REPLY_MAX_LINES,
                fence_text=True,
            )
        )

    def log_params_text(
        self,
        params: Mapping[str, Any],
        name: Optional[str] = None,
        level: LogLevel = LogLevel.DATA,
    ) -> None:
        return

    def publish_preview_snapshot(self, previews: Mapping[str, List[str]]) -> None:
        sections: list[str] = []
        for name, texts in previews.items():
            for index, text in enumerate(texts, start=1):
                item_name = name if len(texts) == 1 else f"{name}/{index}"
                sections.append(f"{item_name}\n{text.strip()}")

        if not sections:
            sections.append("No preview available yet")

        self._safe_post_reply(
            self._format_slack_text(
                "probes",
                "\n\n--------------------------------\n\n".join(sections),
                LogFormat.TEXT,
                LogLevel.INFO,
                max_chars=self._PROBES_MAX_CHARS,
                max_lines=self._PROBES_MAX_LINES,
                fence_text=True,
            )
        )

    def publish_metrics_snapshot(self, steps_ago: int = 0) -> None:
        if len(self._latest_scalars) == 0:
            self._safe_post_reply(self._format_metrics_reply("No metrics available yet"))
            return

        target_step = self._latest_step - steps_ago
        metrics = self._metrics_history.get(target_step)
        if metrics is None:
            self._safe_post_reply(
                self._format_metrics_reply(self._build_metrics_unavailable_text(target_step))
            )
            return

        self._safe_post_reply(self._build_metrics_snapshot_text(target_step, metrics))

    def close(self) -> None:
        self._push_summary(force=True)

    def _push_summary(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_summary_push) < self._notify_interval_sec:
            return

        self._last_summary_push = now

        try:
            self._session.update_root_message(self._build_summary_text())
        except Exception as exc:
            self._warn(f"summary update failed: {exc}")

    def _safe_post_reply(self, text: str) -> None:
        try:
            self._session.post_reply(text)
        except Exception as exc:
            self._warn(f"reply failed: {exc}")

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._warned_at < 60.0:
            return
        self._warned_at = now
        console_log("slack logger", message)

    def _build_summary_text(self) -> str:
        sections: list[str] = []
        progress_lines: list[str] = []

        if self._latest_step > 0:
            progress_lines.append(f"Step: {self._latest_step}")
        else:
            progress_lines.append("Waiting for the first training step")

        last_checkpoint = self._checkpoints.last(with_states=False)
        if last_checkpoint is not None:
            progress_lines.append(f"Last checkpoint: {last_checkpoint.global_step}")

        epoch = self._latest_scalars.get("progress/train_epoch")
        if epoch is not None:
            progress_lines.append(f"Epoch: {int(epoch)}")

        progress_pct = self._latest_scalars.get("progress/pct")
        if progress_pct is not None:
            progress_lines.append(f"Progress: {100.0 * progress_pct:.2f}%")

        if self._fit_started_at is not None:
            elapsed_sec = max(0.0, time.monotonic() - self._fit_started_at)
            progress_lines.append(f"Elapsed: {self._format_elapsed(elapsed_sec)}")

        sections.append(self._format_summary_section("PROGRESS", progress_lines))

        metric_lines = self._tensorboard_preview_lines()
        if metric_lines:
            sections.append(self._format_summary_section("METRICS", metric_lines))

        runtime_lines: list[str] = []
        if "optimizer/learning_rate" in self._latest_scalars:
            runtime_lines.append(f"LR: {self._latest_scalars['optimizer/learning_rate']:.6e}")
        if "perf/units_per_sec" in self._latest_scalars:
            runtime_lines.append(f"Units/sec: {self._latest_scalars['perf/units_per_sec']:.2f}")
        if "progress/examples_seen" in self._latest_scalars:
            runtime_lines.append(f"Examples seen: {int(self._latest_scalars['progress/examples_seen'])}")
        if "progress/units_seen" in self._latest_scalars:
            runtime_lines.append(f"Units seen: {int(self._latest_scalars['progress/units_seen'])}")
        if runtime_lines:
            sections.append(self._format_summary_section("RUNTIME", runtime_lines))

        if self._latest_status:
            sections.append(self._format_summary_section("LATEST STATUS", [self._latest_status]))

        updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return "\n\n".join(
            [
                f"*Run* `{self._run_id}`",
                *sections,
                f"Updated: {updated_at}\n{slack_command_help_hint()}",
            ]
        )

    def _format_summary_section(self, title: str, lines: list[str]) -> str:
        return format_slack_code_block("\n".join([title, *lines]))

    def _build_metrics_snapshot_text(self, step: int, metrics: Mapping[str, float]) -> str:
        lines = [f"Run: {self._run_id}"]
        if step > 0:
            lines.append(f"Step: {step}")
        lines.append(f"Metrics: {len(metrics)}")
        lines.extend(
            f"{name} = {self._format_scalar_value(metrics[name])}"
            for name in sorted(metrics)
        )
        return self._format_metrics_reply("\n".join(lines))

    def _format_metrics_reply(self, text: str) -> str:
        return self._format_slack_text(
            "metrics",
            text,
            LogFormat.TEXT,
            LogLevel.INFO,
            max_chars=self._METRICS_MAX_CHARS,
            max_lines=self._METRICS_MAX_LINES,
            fence_text=True,
        )

    def _build_metrics_unavailable_text(self, target_step: int) -> str:
        retained_steps = sorted(self._metrics_history)
        if not retained_steps:
            return "No metrics available yet"
        return (
            f"Step {target_step} is unavailable; retained steps: "
            f"{retained_steps[0]} through {retained_steps[-1]}"
        )

    def _trim_metrics_history(self) -> None:
        if self._metrics_history_macro_steps < 0:
            return
        while len(self._metrics_history) > self._metrics_history_macro_steps:
            oldest_step = min(self._metrics_history)
            del self._metrics_history[oldest_step]

    def _tensorboard_preview_lines(self) -> list[str]:
        lines: list[str] = []

        for name in ("loss/train", "loss/train_last_batch"):
            value = self._latest_scalars.get(name)
            if value is not None:
                lines.append(f"TB {name}: {value:.6f}")

        for name in sorted(self._latest_scalars):
            if name.startswith("bpb/eval/"):
                lines.append(f"TB {name}: {self._latest_scalars[name]:.6f}")

        return lines

    def _format_scalar_value(self, value: float) -> str:
        if not math.isfinite(value):
            return str(value)

        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return str(int(rounded))

        magnitude = abs(value)
        if magnitude >= 1e5 or (magnitude > 0.0 and magnitude < 1e-4):
            return f"{value:.6e}"

        return f"{value:.6f}".rstrip("0").rstrip(".")

    def _format_elapsed(self, elapsed_sec: float) -> str:
        total_seconds = int(elapsed_sec)
        days, remaining_seconds = divmod(total_seconds, 86_400)
        hours, remaining_seconds = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(remaining_seconds, 60)
        duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{days}d {duration}" if days else duration

    def _format_slack_text(
        self,
        name: str,
        text: str,
        format: LogFormat,
        level: LogLevel,
        *,
        max_chars: int,
        max_lines: int,
        fence_text: bool = False,
    ) -> str:
        compact = self._truncate_for_slack(text.strip(), max_chars=max_chars, max_lines=max_lines)
        prefix = f"[{level.value}] [{name}]"
        if format is LogFormat.JSON:
            return f"{prefix}\n{format_slack_code_block(compact, language='json')}"
        if fence_text:
            return f"{prefix}\n{format_slack_code_block(compact)}"
        if "\n" in compact:
            return f"{prefix}\n{compact}"
        return f"{prefix} {compact}"

    def _truncate_for_slack(self, text: str, *, max_chars: int, max_lines: int) -> str:
        return truncate_for_slack(text, max_chars=max_chars, max_lines=max_lines)
