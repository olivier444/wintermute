from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import torch
from tabulate import tabulate

from wintermute.data.constants import (
    FLD_FILE_ID,
    FLD_FILE_SOURCE_ID,
    FLD_MEMBERS_REC_ID,
    FLD_RECORD_FILE_ID,
    FLD_RECORD_ID,
    FLD_SOURCE_ID,
    FLD_SOURCE_LABEL,
    TBL_FILE,
    TBL_RECORD,
    TBL_SNAPSHOT_MEMBERS,
    TBL_SOURCE,
)
from wintermute.data.iterate.segment_metadata import (
    SEGMENT_COMPONENT_NAME,
    SEGMENT_COMPLETION_LEN,
    SEGMENT_DOC_ID,
    SEGMENT_HAD_HISTORY,
    SEGMENT_HAD_SCRATCHPAD,
    SEGMENT_HAD_SYSTEM_PROMPT,
    SEGMENT_HISTORY_TURNS_DROPPED,
    SEGMENT_KIND,
    SEGMENT_KIND_PROMPT_COMPLETION,
    SEGMENT_PROMPT_LEN,
    SEGMENT_SYSTEM_PROMPT_DROPPED,
    SEGMENT_USED_SCRATCHPAD,
    SEGMENT_WINDOW_END_EXCLUDED,
    SEGMENT_WINDOW_NEW_TOK_START,
    SEGMENT_WINDOW_START,
)
from wintermute.data.iterate.dataclasses import (
    CompositeTrainingViewConfig,
    SnapshotTrainingViewConfig,
    TrainingExample,
    TrainingViewConfig,
)
from wintermute.ml.tasks.factory import TaskWrapperFactory
from wintermute.ml.tasks.implementations.causal_lm.config import CausalLmTaskConfig
from wintermute.ml.runs.manifest import RunManifest
from wintermute.ml.training.config import TrainingConfig
from wintermute.ml.runs.store import RUN_STORE_DIRNAME
from wintermute.tools.files import get_index_dir, get_materialized_dir, json_save, yaml_load
from wintermute.tools.logging import console_log
from wintermute.tools.misc import utc_now
from wintermute.tools.model import iter_batches
from wintermute.tools.tables import build_table_file_path, build_table_sql, create_duckdb_connection


UNCLASSIFIED_VALUE = "<unclassified>"
PROGRESS_LOG_EVERY_SECONDS = 60.0


@dataclass
class _BucketStats:
    examples: int = 0
    segments: int = 0
    total_tokens: int = 0
    loss_only_tokens: int = 0
    packed_examples: int = 0

    def add(self, other: "_BucketStats") -> None:
        self.examples += other.examples
        self.segments += other.segments
        self.total_tokens += other.total_tokens
        self.loss_only_tokens += other.loss_only_tokens
        self.packed_examples += other.packed_examples


@dataclass
class _SftSourceStats:
    segments: int = 0
    segments_with_history: int = 0
    history_reduced: int = 0
    history_turns_dropped_total: int = 0
    segments_with_scratchpad: int = 0
    scratchpad_removed: int = 0
    segments_with_system_prompt: int = 0
    system_prompt_dropped: int = 0

    def add(self, other: "_SftSourceStats") -> None:
        self.segments += other.segments
        self.segments_with_history += other.segments_with_history
        self.history_reduced += other.history_reduced
        self.history_turns_dropped_total += other.history_turns_dropped_total
        self.segments_with_scratchpad += other.segments_with_scratchpad
        self.scratchpad_removed += other.scratchpad_removed
        self.segments_with_system_prompt += other.segments_with_system_prompt
        self.system_prompt_dropped += other.system_prompt_dropped


@dataclass(frozen=True)
class _LoadedAnalyzeRun:
    run_id: str
    training_config: TrainingConfig


class _ProgressReporter:
    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at

    def maybe_report(
        self,
        *,
        overall: _BucketStats,
        by_source_all: Mapping[Tuple[str, str], _BucketStats],
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        elapsed_since_last = now - self.last_report_at

        if not force:
            if elapsed_since_last < PROGRESS_LOG_EVERY_SECONDS:
                return

        total_elapsed = max(now - self.started_at, 1e-9)
        classified_segments = sum(
            stats.segments
            for (src_id, _), stats in by_source_all.items()
            if src_id != UNCLASSIFIED_VALUE
        )
        unclassified_segments = overall.segments - classified_segments
        examples_per_sec = overall.examples / total_elapsed
        tokens_per_sec = overall.total_tokens / total_elapsed

        console_log(
            "analyze",
            (
                f"progress: {overall.examples:_} examples, {overall.segments:_} segments, "
                f"{overall.total_tokens:_} total_tokens, {overall.loss_only_tokens:_} loss_only_tokens, "
                f"{examples_per_sec:.1f} ex/s, {tokens_per_sec:.1f} tok/s, "
                f"{unclassified_segments:_} unclassified segments"
            ),
        )

        self.last_report_at = now


def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):_}"


def _fmt_float(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def _render_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    display_rows = [
        {key.replace("_", " "): value for key, value in row.items()}
        for row in rows
    ]
    return str(tabulate(display_rows, headers="keys", tablefmt="github"))


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


def _iter_snapshot_ids(config: TrainingViewConfig) -> List[str]:
    if isinstance(config, SnapshotTrainingViewConfig):
        return [config.materialized_config.snapshot_id]
    if isinstance(config, CompositeTrainingViewConfig):
        result: List[str] = []
        seen: set[str] = set()
        for source in config.sources:
            for snapshot_id in _iter_snapshot_ids(source.view_config):
                if snapshot_id not in seen:
                    seen.add(snapshot_id)
                    result.append(snapshot_id)
        return result
    raise TypeError(f"Unsupported training view config type: {type(config)!r}")


def _describe_training_view_config(config: TrainingViewConfig) -> str:
    if isinstance(config, SnapshotTrainingViewConfig):
        return config.display_name
    if isinstance(config, CompositeTrainingViewConfig):
        return config.display_name
    return type(config).__name__


def _load_source_lookup(output_root: str, snapshot_id: str) -> Dict[str, Tuple[str, str]]:
    index_path = get_index_dir(output_root)
    members_path = Path(build_table_file_path(TBL_SNAPSHOT_MEMBERS, index_path, snapshot_id))
    if not members_path.exists():
        return {}

    members_table = build_table_sql(TBL_SNAPSHOT_MEMBERS, index_path, snapshot_id)
    record_table = build_table_sql(TBL_RECORD, index_path)
    file_table = build_table_sql(TBL_FILE, index_path)
    source_table = build_table_sql(TBL_SOURCE, index_path)

    con = create_duckdb_connection(temp_directory_root=index_path)

    try:
        rows = con.execute(
            f"""
            SELECT DISTINCT
                m.{FLD_MEMBERS_REC_ID} AS raw_rec_id,
                s.{FLD_SOURCE_ID} AS src_id,
                s.{FLD_SOURCE_LABEL} AS src_label
            FROM {members_table} m
            INNER JOIN {record_table} r ON r.{FLD_RECORD_ID} = m.{FLD_MEMBERS_REC_ID}
            INNER JOIN {file_table} f ON f.{FLD_FILE_ID} = r.{FLD_RECORD_FILE_ID}
            INNER JOIN {source_table} s ON s.{FLD_SOURCE_ID} = f.{FLD_FILE_SOURCE_ID}
            """
        ).fetchall()
    finally:
        con.close()

    return {
        str(raw_rec_id): (str(src_id), str(src_label))
        for raw_rec_id, src_id, src_label in rows
    }


def _resolve_source_info(
    materialized_record_id: str,
    source_lookup: Mapping[str, Tuple[str, str]],
) -> Tuple[str, str]:
    candidate = materialized_record_id
    while True:
        source_info = source_lookup.get(candidate)
        if source_info is not None:
            return source_info
        if "." not in candidate:
            return (UNCLASSIFIED_VALUE, UNCLASSIFIED_VALUE)
        candidate = candidate.rsplit(".", 1)[0]


def _input_ids(example: TrainingExample) -> List[int]:
    payload = dict(example.payload)
    token_ids = payload.get("input_ids")
    if not isinstance(token_ids, list):
        raise RuntimeError(f"Training example '{example.uid}' is missing payload['input_ids'].")
    return [int(x) for x in token_ids]


def _loss_mask(example: TrainingExample) -> List[float]:
    payload = dict(example.payload)
    loss_mask = payload.get("loss_mask")
    if not isinstance(loss_mask, list):
        raise RuntimeError(f"Training example '{example.uid}' is missing payload['loss_mask'].")
    return [float(x) for x in loss_mask]


def _segment_token_stats(segment: Mapping[str, Any]) -> Tuple[int, int] | None:
    if SEGMENT_PROMPT_LEN in segment and SEGMENT_COMPLETION_LEN in segment:
        prompt_len = int(segment[SEGMENT_PROMPT_LEN])
        completion_len = int(segment[SEGMENT_COMPLETION_LEN])
        input_len = prompt_len + completion_len
        loss_len = completion_len
        if prompt_len == 0 and loss_len > 0:
            loss_len -= 1
        return (input_len, loss_len)

    if SEGMENT_WINDOW_START in segment and SEGMENT_WINDOW_END_EXCLUDED in segment and SEGMENT_WINDOW_NEW_TOK_START in segment:
        start = int(segment[SEGMENT_WINDOW_START])
        end_excluded = int(segment[SEGMENT_WINDOW_END_EXCLUDED])
        new_tok_start = int(segment[SEGMENT_WINDOW_NEW_TOK_START])
        input_len = end_excluded - start
        overlap_len = 0 if start == 0 else max(0, min(new_tok_start, end_excluded) - start)
        zero_count = overlap_len if overlap_len > 0 else (1 if input_len > 0 else 0)
        loss_len = max(0, input_len - zero_count)
        return (input_len, loss_len)

    return None


def _example_stats(example: TrainingExample) -> _BucketStats:
    input_len = len(_input_ids(example))
    loss_len = int(sum(_loss_mask(example)))
    segments = example.metadata.get("segments", [])
    segment_count = len(segments) if isinstance(segments, list) and len(segments) > 0 else 1
    return _BucketStats(
        examples=1,
        segments=segment_count,
        total_tokens=input_len,
        loss_only_tokens=loss_len,
        packed_examples=1 if segment_count > 1 else 0,
    )


def _iter_training_budget_examples(
    source: Iterable[TrainingExample],
    *,
    batch_size: int,
    max_training_units: int,
) -> Iterator[TrainingExample]:
    units_seen = 0
    for batch in iter_batches(source, batch_size):
        if units_seen >= max_training_units:
            break

        yield from batch
        units_seen += sum(example.effective_units() for example in batch)


@dataclass(frozen=True)
class _SegmentContribution:
    doc_id: str
    component_name: str | None
    token_count: int
    loss_token_count: int


def _segment_contributions(example: TrainingExample) -> List[_SegmentContribution]:
    total_tokens = len(_input_ids(example))
    total_loss_tokens = int(sum(_loss_mask(example)))
    metadata = dict(example.metadata)
    top_level_component = metadata.get(SEGMENT_COMPONENT_NAME)

    raw_segments = metadata.get("segments", [])
    if not isinstance(raw_segments, list) or len(raw_segments) == 0:
        return [
            _SegmentContribution(
                doc_id=example.uid,
                component_name=str(top_level_component) if top_level_component is not None else None,
                token_count=total_tokens,
                loss_token_count=total_loss_tokens,
            )
        ]

    contributions: List[_SegmentContribution] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            continue
        stats = _segment_token_stats(raw_segment)
        if stats is None:
            continue
        doc_id = str(raw_segment.get(SEGMENT_DOC_ID, example.uid))
        component_name = raw_segment.get(SEGMENT_COMPONENT_NAME, top_level_component)
        contributions.append(
            _SegmentContribution(
                doc_id=doc_id,
                component_name=str(component_name) if component_name is not None else None,
                token_count=stats[0],
                loss_token_count=stats[1],
            )
        )

    if len(contributions) == 0:
        return [
            _SegmentContribution(
                doc_id=example.uid,
                component_name=str(top_level_component) if top_level_component is not None else None,
                token_count=total_tokens,
                loss_token_count=total_loss_tokens,
            )
        ]

    assigned_tokens = sum(item.token_count for item in contributions)
    assigned_loss_tokens = sum(item.loss_token_count for item in contributions)
    token_remainder = total_tokens - assigned_tokens
    loss_remainder = total_loss_tokens - assigned_loss_tokens
    if token_remainder != 0 or loss_remainder != 0:
        first = contributions[0]
        contributions[0] = _SegmentContribution(
            doc_id=first.doc_id,
            component_name=first.component_name,
            token_count=first.token_count + token_remainder,
            loss_token_count=first.loss_token_count + loss_remainder,
        )

    return contributions


def _add_sft_source_diagnostics(
    example: TrainingExample,
    *,
    source_lookup: Mapping[str, Tuple[str, str]],
    by_source: Dict[Tuple[str, str], _SftSourceStats],
) -> None:
    metadata = dict(example.metadata)
    raw_segments = metadata.get("segments", [])
    if not isinstance(raw_segments, list):
        return

    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            continue
        if str(raw_segment.get(SEGMENT_KIND, "")) != SEGMENT_KIND_PROMPT_COMPLETION:
            continue

        doc_id = str(raw_segment.get(SEGMENT_DOC_ID, example.uid))
        source_info = _resolve_source_info(doc_id, source_lookup)
        stats = by_source.setdefault(source_info, _SftSourceStats())
        stats.segments += 1

        had_history = bool(raw_segment.get(SEGMENT_HAD_HISTORY, False))
        history_turns_dropped = int(raw_segment.get(SEGMENT_HISTORY_TURNS_DROPPED, 0))
        if had_history:
            stats.segments_with_history += 1
        if history_turns_dropped > 0:
            stats.history_reduced += 1
            stats.history_turns_dropped_total += history_turns_dropped

        had_scratchpad = bool(raw_segment.get(SEGMENT_HAD_SCRATCHPAD, False))
        used_scratchpad = bool(raw_segment.get(SEGMENT_USED_SCRATCHPAD, False))
        if had_scratchpad:
            stats.segments_with_scratchpad += 1
            if not used_scratchpad:
                stats.scratchpad_removed += 1

        had_system_prompt = bool(raw_segment.get(SEGMENT_HAD_SYSTEM_PROMPT, False))
        system_prompt_dropped = bool(raw_segment.get(SEGMENT_SYSTEM_PROMPT_DROPPED, False))
        if had_system_prompt:
            stats.segments_with_system_prompt += 1
            if system_prompt_dropped:
                stats.system_prompt_dropped += 1


def _sft_stats_to_dict(stats: Any) -> Dict[str, int]:
    return {
        "sft_candidates": int(getattr(stats, "sft_candidates", 0)),
        "sft_emitted": int(getattr(stats, "sft_emitted", 0)),
        "sft_rejected": int(getattr(stats, "sft_rejected", 0)),
        "sft_examples_with_scratchpad": int(getattr(stats, "sft_examples_with_scratchpad", 0)),
        "sft_examples_with_system_prompt": int(getattr(stats, "sft_examples_with_system_prompt", 0)),
        "sft_examples_with_history": int(getattr(stats, "sft_examples_with_history", 0)),
        "sft_scratchpad_removed": int(getattr(stats, "sft_scratchpad_removed", 0)),
        "sft_history_reduced": int(getattr(stats, "sft_history_reduced", 0)),
        "sft_history_turns_dropped_total": int(getattr(stats, "sft_history_turns_dropped_total", 0)),
        "sft_system_prompt_dropped": int(getattr(stats, "sft_system_prompt_dropped", 0)),
        "sft_reject_required_turn_too_long": int(getattr(stats, "sft_reject_required_turn_too_long", 0)),
        "sft_reject_invalid_example": int(getattr(stats, "sft_reject_invalid_example", 0)),
    }


def _format_sft_overview_rows(stats: Mapping[str, int]) -> List[Dict[str, Any]]:
    candidates = max(1, int(stats.get("sft_candidates", 0)))
    scratchpad_examples = max(1, int(stats.get("sft_examples_with_scratchpad", 0)))
    system_prompt_examples = max(1, int(stats.get("sft_examples_with_system_prompt", 0)))
    history_examples = max(1, int(stats.get("sft_examples_with_history", 0)))
    history_turns_dropped_total = int(stats.get("sft_history_turns_dropped_total", 0))
    return [
        {
            "scope": "ALL",
            "candidates": _fmt_int(stats.get("sft_candidates", 0)),
            "emitted": _fmt_int(stats.get("sft_emitted", 0)),
            "rejected": _fmt_int(stats.get("sft_rejected", 0)),
            "rejected_pct": _fmt_pct(100.0 * stats.get("sft_rejected", 0) / candidates),
            "scratchpad_removed": _fmt_int(stats.get("sft_scratchpad_removed", 0)),
            "scratchpad_removed_pct": _fmt_pct(100.0 * stats.get("sft_scratchpad_removed", 0) / candidates),
            "scratchpad_removed_pct_cond": _fmt_pct(100.0 * stats.get("sft_scratchpad_removed", 0) / scratchpad_examples),
            "history_reduced": _fmt_int(stats.get("sft_history_reduced", 0)),
            "history_reduced_pct": _fmt_pct(100.0 * stats.get("sft_history_reduced", 0) / candidates),
            "history_turns_dropped_total": _fmt_int(history_turns_dropped_total),
            "history_turns_dropped_avg": _fmt_float(history_turns_dropped_total / history_examples, digits=2),
            SEGMENT_SYSTEM_PROMPT_DROPPED: _fmt_int(stats.get("sft_system_prompt_dropped", 0)),
            "system_prompt_dropped_pct": _fmt_pct(100.0 * stats.get("sft_system_prompt_dropped", 0) / candidates),
            "system_prompt_dropped_pct_cond": _fmt_pct(100.0 * stats.get("sft_system_prompt_dropped", 0) / system_prompt_examples),
            "reject_turn_too_long": _fmt_int(stats.get("sft_reject_required_turn_too_long", 0)),
        }
    ]


def _build_sft_overview_json_rows(stats: Mapping[str, int]) -> List[Dict[str, Any]]:
    candidates = max(1, int(stats.get("sft_candidates", 0)))
    scratchpad_examples = max(1, int(stats.get("sft_examples_with_scratchpad", 0)))
    system_prompt_examples = max(1, int(stats.get("sft_examples_with_system_prompt", 0)))
    history_examples = max(1, int(stats.get("sft_examples_with_history", 0)))
    history_turns_dropped_total = int(stats.get("sft_history_turns_dropped_total", 0))
    return [
        {
            "scope": "ALL",
            **{key: int(value) for key, value in stats.items()},
            "rejected_pct": 100.0 * stats.get("sft_rejected", 0) / candidates,
            "scratchpad_removed_pct": 100.0 * stats.get("sft_scratchpad_removed", 0) / candidates,
            "scratchpad_removed_pct_of_scratchpad_examples": 100.0 * stats.get("sft_scratchpad_removed", 0) / scratchpad_examples,
            "history_reduced_pct": 100.0 * stats.get("sft_history_reduced", 0) / candidates,
            "history_turns_dropped_avg": history_turns_dropped_total / history_examples,
            "system_prompt_dropped_pct": 100.0 * stats.get("sft_system_prompt_dropped", 0) / candidates,
            "system_prompt_dropped_pct_of_system_prompt_examples": 100.0 * stats.get("sft_system_prompt_dropped", 0) / system_prompt_examples,
        }
    ]


def _format_sft_source_rows(
    rows: Sequence[Tuple[Tuple[str, str], _SftSourceStats]],
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for (src_id, src_label), stats in rows:
        history_segments = max(1, stats.segments_with_history)
        scratchpad_segments = max(1, stats.segments_with_scratchpad)
        system_prompt_segments = max(1, stats.segments_with_system_prompt)
        formatted.append(
            {
                "src_id": src_id,
                "src_label": src_label,
                "sft_segments": _fmt_int(stats.segments),
                "history_reduced": _fmt_int(stats.history_reduced),
                "history_reduced_pct_cond": _fmt_pct(100.0 * stats.history_reduced / history_segments),
                "history_turns_dropped_total": _fmt_int(stats.history_turns_dropped_total),
                "history_turns_dropped_avg": _fmt_float(stats.history_turns_dropped_total / history_segments, digits=2),
                "scratchpad_removed": _fmt_int(stats.scratchpad_removed),
                "scratchpad_removed_pct_cond": _fmt_pct(100.0 * stats.scratchpad_removed / scratchpad_segments),
                SEGMENT_SYSTEM_PROMPT_DROPPED: _fmt_int(stats.system_prompt_dropped),
                "system_prompt_dropped_pct_cond": _fmt_pct(100.0 * stats.system_prompt_dropped / system_prompt_segments),
            }
        )
    return formatted


def _build_sft_source_json_rows(
    rows: Sequence[Tuple[Tuple[str, str], _SftSourceStats]],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for (src_id, src_label), stats in rows:
        history_segments = max(1, stats.segments_with_history)
        scratchpad_segments = max(1, stats.segments_with_scratchpad)
        system_prompt_segments = max(1, stats.segments_with_system_prompt)
        result.append(
            {
                "src_id": src_id,
                "src_label": src_label,
                "sft_segments": stats.segments,
                "segments_with_history": stats.segments_with_history,
                "history_reduced": stats.history_reduced,
                "history_reduced_pct_of_history_segments": 100.0 * stats.history_reduced / history_segments,
                "history_turns_dropped_total": stats.history_turns_dropped_total,
                "history_turns_dropped_avg": stats.history_turns_dropped_total / history_segments,
                "segments_with_scratchpad": stats.segments_with_scratchpad,
                "scratchpad_removed": stats.scratchpad_removed,
                "scratchpad_removed_pct_of_scratchpad_segments": 100.0 * stats.scratchpad_removed / scratchpad_segments,
                "segments_with_system_prompt": stats.segments_with_system_prompt,
                SEGMENT_SYSTEM_PROMPT_DROPPED: stats.system_prompt_dropped,
                "system_prompt_dropped_pct_of_system_prompt_segments": 100.0 * stats.system_prompt_dropped / system_prompt_segments,
            }
        )
    return result


def _format_overview_rows(
    rows: Sequence[Tuple[str, _BucketStats]],
    *,
    source_counts: Mapping[str, int],
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for scope, stats in rows:
        avg_total = stats.total_tokens / stats.examples if stats.examples > 0 else None
        avg_loss_only = stats.loss_only_tokens / stats.examples if stats.examples > 0 else None
        avg_segments = stats.segments / stats.examples if stats.examples > 0 else None
        formatted.append(
            {
                "scope": scope,
                "examples": _fmt_int(stats.examples),
                "segments": _fmt_int(stats.segments),
                "packed_examples": _fmt_int(stats.packed_examples),
                "total_tokens": _fmt_int(stats.total_tokens),
                "loss_only_tokens": _fmt_int(stats.loss_only_tokens),
                "loss_token_ratio": _fmt_pct(
                    100.0 * stats.loss_only_tokens / stats.total_tokens if stats.total_tokens > 0 else None
                ),
                "avg_total": _fmt_float(avg_total, digits=1),
                "avg_loss_only": _fmt_float(avg_loss_only, digits=1),
                "avg_segments": _fmt_float(avg_segments, digits=2),
                "srcs": _fmt_int(source_counts.get(scope, 0)),
            }
        )
    return formatted


def _build_overview_json_rows(
    rows: Sequence[Tuple[str, _BucketStats]],
    *,
    source_counts: Mapping[str, int],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for scope, stats in rows:
        result.append(
            {
                "scope": scope,
                "examples": stats.examples,
                "segments": stats.segments,
                "packed_examples": stats.packed_examples,
                "total_tokens": stats.total_tokens,
                "loss_only_tokens": stats.loss_only_tokens,
                "loss_token_ratio_pct": (
                    100.0 * stats.loss_only_tokens / stats.total_tokens
                    if stats.total_tokens > 0
                    else None
                ),
                "avg_total_tokens_per_example": (
                    stats.total_tokens / stats.examples
                    if stats.examples > 0
                    else None
                ),
                "avg_loss_only_tokens_per_example": (
                    stats.loss_only_tokens / stats.examples
                    if stats.examples > 0
                    else None
                ),
                "avg_segments_per_example": (
                    stats.segments / stats.examples
                    if stats.examples > 0
                    else None
                ),
                "source_count": source_counts.get(scope, 0),
            }
        )
    return result


def _format_source_rows(
    rows: Sequence[Tuple[Tuple[str, str], _BucketStats]],
    *,
    scope_loss_only_tokens: int,
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for (src_id, src_label), stats in rows:
        avg_total = stats.total_tokens / stats.segments if stats.segments > 0 else None
        avg_loss_only = stats.loss_only_tokens / stats.segments if stats.segments > 0 else None
        formatted.append(
            {
                "src_id": src_id,
                "src_label": src_label,
                "segments": _fmt_int(stats.segments),
                "total_tokens": _fmt_int(stats.total_tokens),
                "loss_only_tokens": _fmt_int(stats.loss_only_tokens),
                "loss_token_share": _fmt_pct(
                    100.0 * stats.loss_only_tokens / scope_loss_only_tokens if scope_loss_only_tokens else None
                ),
                "loss_token_ratio": _fmt_pct(
                    100.0 * stats.loss_only_tokens / stats.total_tokens if stats.total_tokens > 0 else None
                ),
                "avg_total": _fmt_float(avg_total, digits=1),
                "avg_loss_only": _fmt_float(avg_loss_only, digits=1),
            }
        )
    return formatted


def _build_source_json_rows(
    rows: Sequence[Tuple[Tuple[str, str], _BucketStats]],
    *,
    scope_loss_only_tokens: int,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for (src_id, src_label), stats in rows:
        result.append(
            {
                "src_id": src_id,
                "src_label": src_label,
                "segments": stats.segments,
                "total_tokens": stats.total_tokens,
                "loss_only_tokens": stats.loss_only_tokens,
                "loss_token_share_pct": (
                    100.0 * stats.loss_only_tokens / scope_loss_only_tokens
                    if scope_loss_only_tokens > 0
                    else None
                ),
                "loss_token_ratio_pct": (
                    100.0 * stats.loss_only_tokens / stats.total_tokens
                    if stats.total_tokens > 0
                    else None
                ),
                "avg_total_tokens_per_segment": (
                    stats.total_tokens / stats.segments
                    if stats.segments > 0
                    else None
                ),
                "avg_loss_only_tokens_per_segment": (
                    stats.loss_only_tokens / stats.segments
                    if stats.segments > 0
                    else None
                ),
            }
        )
    return result


def _format_component_rows(
    rows: Sequence[Tuple[str, _BucketStats]],
    *,
    scope_loss_only_tokens: int,
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for component_path, stats in rows:
        avg_total = stats.total_tokens / stats.segments if stats.segments > 0 else None
        avg_loss_only = stats.loss_only_tokens / stats.segments if stats.segments > 0 else None
        formatted.append(
            {
                "component": component_path,
                "segments": _fmt_int(stats.segments),
                "total_tokens": _fmt_int(stats.total_tokens),
                "loss_only_tokens": _fmt_int(stats.loss_only_tokens),
                "loss_token_share": _fmt_pct(
                    100.0 * stats.loss_only_tokens / scope_loss_only_tokens if scope_loss_only_tokens else None
                ),
                "loss_token_ratio": _fmt_pct(
                    100.0 * stats.loss_only_tokens / stats.total_tokens if stats.total_tokens > 0 else None
                ),
                "avg_total": _fmt_float(avg_total, digits=1),
                "avg_loss_only": _fmt_float(avg_loss_only, digits=1),
            }
        )
    return formatted


def _build_component_json_rows(
    rows: Sequence[Tuple[str, _BucketStats]],
    *,
    scope_loss_only_tokens: int,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for component_path, stats in rows:
        result.append(
            {
                "component": component_path,
                "segments": stats.segments,
                "total_tokens": stats.total_tokens,
                "loss_only_tokens": stats.loss_only_tokens,
                "loss_token_share_pct": (
                    100.0 * stats.loss_only_tokens / scope_loss_only_tokens
                    if scope_loss_only_tokens > 0
                    else None
                ),
                "loss_token_ratio_pct": (
                    100.0 * stats.loss_only_tokens / stats.total_tokens
                    if stats.total_tokens > 0
                    else None
                ),
                "avg_total_tokens_per_segment": (
                    stats.total_tokens / stats.segments
                    if stats.segments > 0
                    else None
                ),
                "avg_loss_only_tokens_per_segment": (
                    stats.loss_only_tokens / stats.segments
                    if stats.segments > 0
                    else None
                ),
            }
        )
    return result


def _build_ascii_report(
    *,
    config_name: str,
    top_sources: int,
    overview_rows: List[Dict[str, Any]],
    source_rows: List[Dict[str, Any]],
    component_rows: List[Dict[str, Any]] | None,
    sft_overview_rows: List[Dict[str, Any]] | None,
    sft_source_rows: List[Dict[str, Any]] | None,
) -> str:
    sections = [
        "=== Overview ===\n" + _render_table(overview_rows),
    ]

    if component_rows is not None:
        sections.append(
            (
                f"=== Per Composite Component ({config_name}, "
                f"top {top_sources if top_sources > 0 else 'all'} by loss_only_tokens) ===\n"
                f"{_render_table(component_rows)}"
            )
        )

    sections.append(
        (
            f"=== Per Source ({config_name}, "
            f"top {top_sources if top_sources > 0 else 'all'} by loss_only_tokens) ===\n"
            f"{_render_table(source_rows)}"
        )
    )
    return "\n\n".join(sections) + "\n"


def _write_analyze_artifacts(
    *,
    output_dir: Path,
    file_stem: str,
    json_payload: Dict[str, Any],
    ascii_report: str,
) -> Tuple[Path, Path]:
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
) -> None:
    if top_sources == 0:
        raise ValueError("top_sources cannot be 0; use a negative value to show all rows")

    loaded_run = _load_analyze_run(training_config_path)
    training_config = loaded_run.training_config
    task_config = CausalLmTaskConfig.from_dict(training_config.task_config.params)
    training_view_config = training_config.training_view_config
    config_name = _describe_training_view_config(training_view_config)
    tokenizer_label = (
        task_config.tokenizer_id
        or task_config.tokenizer_file
        or task_config.hf_tokenizer_name
        or "<unknown>"
    )
    max_example_len = int(task_config.windowing_policy.window_max_len)
    add_eos_at_end_of_doc = bool(task_config.windowing_policy.add_eos_at_end_of_doc)
    max_training_units = training_config.trainer_config.max_training_units
    batch_size = training_config.trainer_config.batch_size

    source_lookup: Dict[str, Tuple[str, str]] = {}
    for snapshot_id in _iter_snapshot_ids(training_view_config):
        source_lookup.update(_load_source_lookup(output_root, snapshot_id))

    overall = _BucketStats()
    by_source_all: Dict[Tuple[str, str], _BucketStats] = {}
    by_component_all: Dict[str, _BucketStats] = {}
    sft_by_source_all: Dict[Tuple[str, str], _SftSourceStats] = {}
    progress = _ProgressReporter()

    wrapper = TaskWrapperFactory().build_wrapper(
        training_config.task_config,
        device=torch.device("cpu"),
        bf_16_enabled=False,
        root_path=output_root,
    )
    training_view = wrapper.build_training_view(
        output_root=output_root,
        training_view_config=training_view_config,
        silent=False,
    )

    console_log(
        "analyze",
        f"scanning training view: {config_name} (max_training_units={max_training_units:_})",
    )
    for example in _iter_training_budget_examples(
        training_view,
        batch_size=batch_size,
        max_training_units=max_training_units,
    ):
        example_stats = _example_stats(example)
        overall.add(example_stats)

        for contribution in _segment_contributions(example):
            source_info = _resolve_source_info(contribution.doc_id, source_lookup)
            segment_stats = _BucketStats(
                segments=1,
                total_tokens=contribution.token_count,
                loss_only_tokens=contribution.loss_token_count,
            )
            by_source_all.setdefault(source_info, _BucketStats()).add(segment_stats)
            if contribution.component_name is not None:
                by_component_all.setdefault(contribution.component_name, _BucketStats()).add(segment_stats)

        _add_sft_source_diagnostics(
            example,
            source_lookup=source_lookup,
            by_source=sft_by_source_all,
        )

        progress.maybe_report(overall=overall, by_source_all=by_source_all)

    progress.maybe_report(overall=overall, by_source_all=by_source_all, force=True)

    console_log("analyze", f"training_config: {training_config_path}")
    console_log("analyze", f"training_view_config: {config_name}")
    console_log("analyze", f"output_root: {output_root}")
    console_log("analyze", f"tokenizer: {tokenizer_label}")
    console_log("analyze", f"add_eos_at_end_of_doc: {add_eos_at_end_of_doc}")
    console_log(
        "analyze",
        f"training budget: {overall.loss_only_tokens:_} / {max_training_units:_} loss tokens",
    )
    console_log(
        "analyze",
        f"token counting is runtime-exact on emitted training examples; max_example_len={max_example_len}.",
    )

    training_view_stats = _sft_stats_to_dict(getattr(training_view, "last_iteration_stats", None))
    overview_rows: List[Tuple[str, _BucketStats]] = [("ALL", overall)]
    source_counts = {"ALL": len(by_source_all)}
    overview_table_rows = _format_overview_rows(
        overview_rows,
        source_counts=source_counts,
    )
    overview_json_rows = _build_overview_json_rows(
        overview_rows,
        source_counts=source_counts,
    )

    source_limit = None if top_sources < 0 else int(top_sources)
    component_table_rows: List[Dict[str, Any]] | None = None
    component_json_rows: List[Dict[str, Any]] | None = None
    sft_overview_table_rows: List[Dict[str, Any]] | None = None
    sft_overview_json_rows: List[Dict[str, Any]] | None = None
    sft_source_table_rows: List[Dict[str, Any]] | None = None
    sft_source_json_rows: List[Dict[str, Any]] | None = None
    if isinstance(training_view_config, CompositeTrainingViewConfig):
        component_rows = sorted(
            by_component_all.items(),
            key=lambda item: (-item[1].loss_only_tokens, -item[1].examples, item[0]),
        )
        if source_limit is not None:
            component_rows = component_rows[:source_limit]
        component_table_rows = _format_component_rows(
            component_rows,
            scope_loss_only_tokens=overall.loss_only_tokens,
        )
        component_json_rows = _build_component_json_rows(
            component_rows,
            scope_loss_only_tokens=overall.loss_only_tokens,
        )

    if training_view_stats.get("sft_candidates", 0) > 0:
        sft_overview_table_rows = _format_sft_overview_rows(training_view_stats)
        sft_overview_json_rows = _build_sft_overview_json_rows(training_view_stats)

        sft_source_rows_all = sorted(
            sft_by_source_all.items(),
            key=lambda item: (
                -item[1].history_turns_dropped_total,
                -item[1].history_reduced,
                -item[1].scratchpad_removed,
                -item[1].system_prompt_dropped,
                -item[1].segments,
                item[0][0],
            ),
        )
        if source_limit is not None:
            sft_source_rows_all = sft_source_rows_all[:source_limit]
        sft_source_table_rows = _format_sft_source_rows(sft_source_rows_all)
        sft_source_json_rows = _build_sft_source_json_rows(sft_source_rows_all)

    all_source_rows = sorted(
        by_source_all.items(),
        key=lambda item: (-item[1].loss_only_tokens, -item[1].examples, item[0][0]),
    )
    if source_limit is not None:
        all_source_rows = all_source_rows[:source_limit]
    source_table_rows = _format_source_rows(
        all_source_rows,
        scope_loss_only_tokens=overall.loss_only_tokens,
    )
    source_json_rows = _build_source_json_rows(
        all_source_rows,
        scope_loss_only_tokens=overall.loss_only_tokens,
    )

    ascii_report = _build_ascii_report(
        config_name=config_name,
        top_sources=top_sources,
        overview_rows=overview_table_rows,
        source_rows=source_table_rows,
        component_rows=component_table_rows,
        sft_overview_rows=sft_overview_table_rows,
        sft_source_rows=sft_source_table_rows,
    )
    print(ascii_report, end="")

    generated_at = utc_now()
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
    file_stem = f"stats_{loaded_run.run_id}_{timestamp}"
    json_payload: Dict[str, Any] = {
        "run_id": loaded_run.run_id,
        "training_config_path": training_config_path,
        "output_root": output_root,
        "generated_at_utc": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "top_sources": top_sources,
        "train_view_config_name": config_name,
        "tokenizer": tokenizer_label,
        "add_eos_at_end_of_doc": add_eos_at_end_of_doc,
        "max_example_len": max_example_len,
        "max_training_units": max_training_units,
        "overall": asdict(overall),
        "overview_rows": overview_json_rows,
        "source_rows": source_json_rows,
        "component_rows": component_json_rows,
        "sft_context_overview_rows": sft_overview_json_rows,
        "sft_context_source_rows": sft_source_json_rows,
        "sft_context_stats": training_view_stats if training_view_stats.get("sft_candidates", 0) > 0 else None,
    }

    run_store_root = Path(output_root) / RUN_STORE_DIRNAME
    run_json_path, run_txt_path = _write_analyze_artifacts(
        output_dir=run_store_root,
        file_stem=file_stem,
        json_payload=json_payload,
        ascii_report=ascii_report,
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
            ascii_report=ascii_report,
        )
        console_log("analyze", f"saved analyze artifacts: {materialized_json_path}")
        console_log("analyze", f"saved analyze artifacts: {materialized_txt_path}")
