from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import yaml

from wintermute.ml.evaluation.lm_eval.config import LmEvalConfig
from wintermute.ml.evaluation.lm_eval.harness import LmEvalHarness
from wintermute.ml.evaluation.lm_eval.report import LmEvalReport
from wintermute.ml.evaluation.lm_eval.targets import (
    LmEvalTarget,
    ResolvedLmEvalTarget,
    resolve_lm_eval_target,
)
from wintermute.ml.runs import ResolvedCheckpoint
from wintermute.tools.logging import console_log
from wintermute.tools.misc import utc_now


RESULTS_FILENAME = "results.json"
RESULTS_REPORT_FILENAME = "results.txt"
LM_EVAL_CONFIG_FILENAME = "lm_eval_config.yaml"


@dataclass(frozen=True)
class LmEvalResult:
    output_dir: Path
    results_path: Path
    report_path: Path
    checkpoint: ResolvedCheckpoint | None
    results: Mapping[str, Any]


@dataclass(frozen=True)
class LmEvalExecution:
    output_root: Path
    config: LmEvalConfig

    @classmethod
    def from_yaml(
        cls,
        output_root: str | Path,
        config_path: str | Path,
    ) -> "LmEvalExecution":
        return cls(
            output_root=Path(output_root),
            config=LmEvalConfig.load(config_path),
        )

    def run(self, target: LmEvalTarget) -> LmEvalResult:
        resolved_target = resolve_lm_eval_target(
            self.output_root,
            target,
            self.config.values,
            self.config.device,
        )
        return self.run_resolved(resolved_target)

    def run_resolved(self, resolved_target: ResolvedLmEvalTarget) -> LmEvalResult:
        adapter, collector = self._traced_adapter(resolved_target)
        harness_result = LmEvalHarness(self.output_root).evaluate(adapter, self.config)
        return self._write_artifacts(
            resolved_target,
            harness_result.results,
            harness_result.cache_path,
            collector,
        )

    def _traced_adapter(self, target: ResolvedLmEvalTarget) -> tuple[Any, Any | None]:
        if self.config.sample_rate == 0.0:
            return target.adapter, None
        from wintermute.ml.evaluation.lm_eval.traces import LmEvalTraceCollector, TracedLmEvalAdapter

        collector = LmEvalTraceCollector(self.config.sample_rate)
        return TracedLmEvalAdapter(target.adapter, collector), collector

    def _write_artifacts(
        self,
        target: ResolvedLmEvalTarget,
        results: Mapping[str, Any],
        cache_path: Path,
        collector: Any | None,
    ) -> LmEvalResult:
        generated_at = utc_now()
        output_dir = target.evaluation_root / generated_at.strftime("%Y%m%d_%H%M%S_%f")
        output_dir.mkdir(parents=True, exist_ok=False)
        config_artifact_path = output_dir / LM_EVAL_CONFIG_FILENAME
        if self.config.path is not None:
            shutil.copy2(self.config.path, config_artifact_path)
        else:
            config_artifact_path.write_text(
                yaml.safe_dump(dict(self.config.values), sort_keys=False),
                encoding="utf-8",
            )
        payload = dict(results)
        payload["wintermute"] = {
            "engine": "lm-evaluation-harness",
            "engine_version": _lm_eval_version(),
            "generated_at_utc": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            **target.metadata,
            "lm_eval_config_path": (
                self.config.path.as_posix()
                if self.config.path is not None
                else None
            ),
            "lm_eval_cache_path": cache_path.as_posix(),
            "device": str(self.config.device),
        }
        if collector is not None:
            from wintermute.ml.evaluation.lm_eval.traces import SAMPLES_REPORT_FILENAME

            samples_report_path = output_dir / SAMPLES_REPORT_FILENAME
            collector.write_report(samples_report_path)
            payload["wintermute"].update(
                sample_rate=self.config.sample_rate,
                sample_trace_count=len(collector.traces),
                samples_report=samples_report_path.name,
            )

        report = LmEvalReport(payload)
        results_path = output_dir / RESULTS_FILENAME
        report.write_json(results_path)
        report_path = output_dir / RESULTS_REPORT_FILENAME
        report.write_text(report_path)
        console_log(
            "eval",
            f"results written to {results_path.as_posix()} and {report_path.as_posix()}",
        )
        return LmEvalResult(
            output_dir=output_dir,
            results_path=results_path,
            report_path=report_path,
            checkpoint=target.checkpoint,
            results=results,
        )


def _lm_eval_version() -> str:
    try:
        return version("lm_eval")
    except PackageNotFoundError:
        return "unknown"
