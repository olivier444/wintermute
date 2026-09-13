from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from tabulate import tabulate

from wintermute.tools.misc import json_default


@dataclass(frozen=True)
class LmEvalReport:
    """Structured lm-eval results with renderers for persisted report formats."""

    payload: Mapping[str, Any]

    def write_json(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )

    def write_text(self, path: Path) -> None:
        path.write_text(self.render_text(), encoding="utf-8")

    def render_text(self) -> str:
        wintermute = self._mapping("wintermute")
        return "\n".join(
            [
                "=== lm-evaluation-harness results ===",
                f"Target: {self._target_label(wintermute)}",
                f"Checkpoint: {wintermute.get('checkpoint_id', 'n/a')}",
                f"Global step: {wintermute.get('checkpoint_global_step', 'n/a')}",
                f"Training units seen: {wintermute.get('checkpoint_units_seen', 'n/a')}",
                f"Device: {wintermute.get('device', 'n/a')}",
                f"Generated at (UTC): {wintermute.get('generated_at_utc', 'n/a')}",
                "",
                "=== Metrics ===",
                self._render_metrics(self._mapping("results")),
                "",
            ]
        )

    def _mapping(self, key: str) -> Mapping[str, Any]:
        value = self.payload.get(key, {})
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _target_label(wintermute: Mapping[str, Any]) -> str:
        if wintermute.get("target_type") == "huggingface_model":
            return f"Hugging Face model {wintermute.get('model_name', 'n/a')}"
        return f"Run {wintermute.get('run_id', 'n/a')}"

    @staticmethod
    def _render_metrics(results: Mapping[str, Any]) -> str:
        rows: list[dict[str, str]] = []
        for task_name, raw_metrics in sorted(results.items()):
            if not isinstance(raw_metrics, Mapping):
                continue
            for metric_key, raw_value in sorted(raw_metrics.items()):
                if not isinstance(raw_value, Real) or metric_key.endswith("_stderr,none"):
                    continue
                if "," in metric_key:
                    metric_name, filter_name = metric_key.split(",", maxsplit=1)
                    stderr_key = f"{metric_name}_stderr,{filter_name}"
                else:
                    metric_name = metric_key
                    stderr_key = f"{metric_name}_stderr"
                raw_stderr = raw_metrics.get(stderr_key)
                rows.append(
                    {
                        "Task": str(task_name),
                        "Metric": metric_name,
                        "Score": LmEvalReport._format_metric(metric_name, float(raw_value)),
                        "Std. error": (
                            LmEvalReport._format_metric(metric_name, float(raw_stderr))
                            if isinstance(raw_stderr, Real)
                            else "n/a"
                        ),
                    }
                )
        if not rows:
            return "(no numeric task metrics)"
        return str(tabulate(rows, headers="keys", tablefmt="github"))

    @staticmethod
    def _format_metric(metric_name: str, value: float) -> str:
        is_score = metric_name.lower().startswith(
            ("acc", "exact_match", "em", "f1", "precision", "recall")
        )
        if is_score and 0.0 <= value <= 1.0:
            return f"{value * 100:.2f}%"
        return f"{value:.6g}"


def _json_default(value: Any) -> Any:
    scalar = getattr(value, "item", None)
    if callable(scalar):
        return scalar()
    if isinstance(value, set):
        return list(value)
    return json_default(value)
