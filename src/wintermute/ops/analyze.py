from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from wintermute.data.iterate.dataclasses import (
    CompositeTrainingViewConfig,
    SnapshotTrainingViewConfig,
    TrainingViewConfig,
)
from wintermute.ml.runs.manifest import RunManifest
from wintermute.ml.runs.store import RUN_STORE_DIRNAME
from wintermute.ml.tasks.factory import TaskWrapperFactory
from wintermute.ml.tasks.implementations.causal_lm.analysis import (
    CompositeCapacityEstimator,
    load_training_view_source_lookup,
    scan_training_view,
)
from wintermute.ml.tasks.implementations.causal_lm.analysis_report import (
    build_training_view_analysis_report,
)
from wintermute.ml.tasks.implementations.causal_lm.config import CausalLmTaskConfig
from wintermute.ml.training.config import RunConfig
from wintermute.tools.files import get_materialized_dir, json_save, yaml_load
from wintermute.tools.logging import console_log
from wintermute.tools.misc import utc_now


@dataclass(frozen=True)
class _LoadedAnalyzeRun:
    run_id: str
    training_config: RunConfig


def _load_analyze_run(training_config_path: str) -> _LoadedAnalyzeRun:
    config_path = Path(training_config_path)
    payload = yaml_load(config_path.parent.as_posix(), config_path.name)
    if not isinstance(payload, Mapping):
        raise ValueError("training config payload must be a YAML mapping")

    manifest_payload = dict(payload)
    configuration = manifest_payload.get("configuration")
    if isinstance(configuration, Mapping):
        configuration = dict(configuration)
        configuration.pop("slack_config", None)
        manifest_payload["configuration"] = configuration
    else:
        manifest_payload.pop("slack_config", None)

    manifest = RunManifest.from_dict(manifest_payload)
    return _LoadedAnalyzeRun(
        run_id=manifest.run_id,
        training_config=manifest.config,
    )


def _describe_training_view_config(config: TrainingViewConfig) -> str:
    if isinstance(config, (SnapshotTrainingViewConfig, CompositeTrainingViewConfig)):
        return config.display_name
    return type(config).__name__


def _write_analyze_artifacts(
    *,
    output_dir: Path,
    file_stem: str,
    json_payload: dict[str, Any],
    ascii_report: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{file_stem}.json"
    txt_path = output_dir / f"{file_stem}.txt"
    json_save(json_payload, output_dir.as_posix(), json_path.name)
    txt_path.write_text(ascii_report, encoding="utf-8")
    return (json_path, txt_path)


def run_analyze_training_config(
    *,
    output_root: str,
    training_config_path: str,
    top_sources: int = -1,
    mode: str = "fast",
) -> None:
    if top_sources == 0:
        raise ValueError("top_sources cannot be 0; use a negative value to show all rows")
    if mode not in {"fast", "full"}:
        raise ValueError("mode must be 'fast' or 'full'")

    loaded_run = _load_analyze_run(training_config_path)
    training_config = loaded_run.training_config
    task_config = CausalLmTaskConfig.from_dict(training_config.task_config.params)
    training_view_config = training_config.training_view_config
    config_name = _describe_training_view_config(training_view_config)
    max_training_units = training_config.trainer_config.max_training_units
    capacity_estimator = _build_capacity_estimator(
        mode=mode,
        training_view_config=training_view_config,
        max_training_units=max_training_units,
    )
    source_lookup = load_training_view_source_lookup(
        output_root,
        training_view_config,
    )

    wrapper = TaskWrapperFactory().build_wrapper(
        training_config.task_config,
        device=torch.device("cpu"),
        bf_16_enabled=False,
        root_path=output_root,
    )
    training_view = wrapper.build_training_view(
        output_root=output_root,
        training_view_config=training_view_config,
        source_exhaustion_observer=(
            capacity_estimator.observe_exhaustion
            if capacity_estimator is not None
            else None
        ),
        silent=False,
    )

    console_log(
        "analyze",
        f"scanning training view: {config_name} (mode={mode}, max_training_units={max_training_units:_})",
    )
    analysis = scan_training_view(
        training_view,
        source_lookup=source_lookup,
        batch_size=training_config.trainer_config.batch_size,
        max_training_units=max_training_units,
        capacity_estimator=capacity_estimator,
    )
    _log_fast_fail(capacity_estimator, max_training_units=max_training_units)

    tokenizer_label = (
        task_config.tokenizer_id
        or task_config.tokenizer_file
        or task_config.hf_tokenizer_name
        or "<unknown>"
    )
    max_example_len = int(task_config.windowing_policy.window_max_len)
    add_eos_at_end_of_doc = bool(task_config.windowing_policy.add_eos_at_end_of_doc)
    console_log("analyze", f"training_config: {training_config_path}")
    console_log("analyze", f"mode: {mode}")
    console_log("analyze", f"training_view_config: {config_name}")
    console_log("analyze", f"output_root: {output_root}")
    console_log("analyze", f"tokenizer: {tokenizer_label}")
    console_log("analyze", f"add_eos_at_end_of_doc: {add_eos_at_end_of_doc}")
    console_log(
        "analyze",
        f"training budget: {analysis.overall.loss_only_tokens:_} / {max_training_units:_} loss tokens",
    )
    console_log(
        "analyze",
        f"token counting is runtime-exact on emitted training examples; max_example_len={max_example_len}.",
    )

    report = build_training_view_analysis_report(
        analysis,
        config_name=config_name,
        is_composite=isinstance(training_view_config, CompositeTrainingViewConfig),
        top_sources=top_sources,
        max_training_units=max_training_units,
        capacity_estimator=capacity_estimator,
    )
    print(report.ascii_report, end="")

    generated_at = utc_now()
    file_stem = f"stats_{loaded_run.run_id}_{generated_at.strftime('%Y%m%d_%H%M%S')}"
    json_payload: dict[str, Any] = {
        "run_id": loaded_run.run_id,
        "training_config_path": training_config_path,
        "output_root": output_root,
        "generated_at_utc": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "top_sources": top_sources,
        "mode": mode,
        "train_view_config_name": config_name,
        "tokenizer": tokenizer_label,
        "add_eos_at_end_of_doc": add_eos_at_end_of_doc,
        "max_example_len": max_example_len,
        "max_training_units": max_training_units,
        "overall": asdict(analysis.overall),
        "overview_rows": report.overview_rows,
        "source_rows": report.source_rows,
        "component_rows": report.component_rows,
        "sft_context_overview_rows": report.sft_context_overview_rows,
        "sft_context_source_rows": report.sft_context_source_rows,
        "sft_context_stats": (
            analysis.sft_context_stats
            if analysis.sft_context_stats.get("sft_candidates", 0) > 0
            else None
        ),
        "fast_capacity_result": capacity_estimator.result if capacity_estimator is not None else None,
        "fast_capacity_rows": (
            [asdict(source) for source in capacity_estimator.sources]
            if capacity_estimator is not None
            else None
        ),
    }

    run_json_path, run_txt_path = _write_analyze_artifacts(
        output_dir=Path(output_root) / RUN_STORE_DIRNAME,
        file_stem=file_stem,
        json_payload=json_payload,
        ascii_report=report.ascii_report,
    )
    console_log("analyze", f"saved analyze artifacts: {run_json_path}")
    console_log("analyze", f"saved analyze artifacts: {run_txt_path}")

    if isinstance(training_view_config, SnapshotTrainingViewConfig):
        materialized_dir = Path(
            get_materialized_dir(
                output_root,
                training_view_config.materialized_config.snapshot_id,
                training_view_config.materialized_config.split,
            )
        )
        materialized_json_path, materialized_txt_path = _write_analyze_artifacts(
            output_dir=materialized_dir,
            file_stem=file_stem,
            json_payload=json_payload,
            ascii_report=report.ascii_report,
        )
        console_log("analyze", f"saved analyze artifacts: {materialized_json_path}")
        console_log("analyze", f"saved analyze artifacts: {materialized_txt_path}")


def _build_capacity_estimator(
    *,
    mode: str,
    training_view_config: TrainingViewConfig,
    max_training_units: int,
) -> CompositeCapacityEstimator | None:
    if mode != "fast":
        return None
    if not isinstance(training_view_config, CompositeTrainingViewConfig):
        console_log(
            "analyze",
            "fast capacity estimation requires a composite training view; falling back to a full scan",
        )
        return None
    if training_view_config.mix_unit != "loss_tokens":
        console_log(
            "analyze",
            "fast capacity estimation requires mix_unit=loss_tokens; falling back to a full scan",
        )
        return None
    return CompositeCapacityEstimator(
        training_view_config,
        max_training_units,
    )


def _log_fast_fail(
    estimator: CompositeCapacityEstimator | None,
    *,
    max_training_units: int,
) -> None:
    if estimator is None or estimator.limiting_source is None:
        return
    source = estimator.limiting_source
    assert source.estimated_max_training_units is not None
    console_log(
        "analyze",
        (
            f"fast fail: source '{source.name}' projects only "
            f"{source.estimated_max_training_units:_.0f} training units, "
            f"below the configured {max_training_units:_}"
        ),
    )
