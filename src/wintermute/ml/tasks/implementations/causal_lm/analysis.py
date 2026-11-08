from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping

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
from wintermute.data.iterate.dataclasses import (
    CompositeTrainingViewConfig,
    SnapshotTrainingViewConfig,
    TrainingExample,
    TrainingViewConfig,
)
from wintermute.data.iterate.segment_metadata import (
    SEGMENT_COMPLETION_LEN,
    SEGMENT_COMPONENT_NAME,
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
from wintermute.tools.files import get_index_dir
from wintermute.tools.logging import console_log
from wintermute.tools.model import iter_batches
from wintermute.tools.tables import (
    build_table_file_path,
    build_table_sql,
    create_duckdb_connection,
)


UNCLASSIFIED_VALUE = "<unclassified>"
PROGRESS_LOG_EVERY_SECONDS = 60.0
SourceInfo = tuple[str, str]
CapacityStatus = Literal["undecided", "probably_valid", "limiting"]


@dataclass
class SourceCapacity:
    name: str
    weight: float
    allowed_passes: int
    status: CapacityStatus = "undecided"
    completed_passes: int = 0
    estimated_max_training_units: float | None = None


class CompositeCapacityEstimator:
    def __init__(
        self,
        config: CompositeTrainingViewConfig,
        max_training_units: int,
    ) -> None:
        total_weight = sum(source.weight for source in config.sources)
        if total_weight <= 0:
            raise ValueError("composite training view must have a positive total source weight")
        self.max_training_units = max_training_units
        self.sources = [
            SourceCapacity(
                name=source.display_name,
                weight=source.weight / total_weight,
                allowed_passes=source.max_restarts + 1,
                status="probably_valid" if source.weight <= 0 else "undecided",
            )
            for source in config.sources
        ]
        self.limiting_source: SourceCapacity | None = None
        self.budget_reached = False

    def observe_exhaustion(
        self,
        source_index: int,
        completed_passes: int,
        source_units: int,
        composite_units: int,
    ) -> bool:
        source = self.sources[source_index]
        source.completed_passes = completed_passes
        if source.weight <= 0:
            return False

        source.estimated_max_training_units = (
            float(composite_units)
            if completed_passes >= source.allowed_passes
            else source_units / completed_passes * source.allowed_passes / source.weight
        )
        source.status = (
            "limiting"
            if source.estimated_max_training_units < self.max_training_units
            else "probably_valid"
        )
        if source.status == "limiting":
            self.limiting_source = source
            return True
        return not self.undecided_source_names()

    def undecided_source_names(self) -> list[str]:
        return [source.name for source in self.sources if source.status == "undecided"]

    def mark_budget_reached(self) -> None:
        self.budget_reached = True
        self.limiting_source = None
        for source in self.sources:
            source.status = "probably_valid"

    @property
    def result(self) -> str:
        if self.limiting_source is not None:
            return "insufficient_estimated_capacity"
        if self.budget_reached:
            return "budget_reached"
        if not self.undecided_source_names():
            return "sufficient_estimated_capacity"
        return "view_exhausted"


@dataclass
class BucketStats:
    examples: int = 0
    segments: int = 0
    total_tokens: int = 0
    loss_only_tokens: int = 0
    packed_examples: int = 0

    def add(self, other: BucketStats) -> None:
        self.examples += other.examples
        self.segments += other.segments
        self.total_tokens += other.total_tokens
        self.loss_only_tokens += other.loss_only_tokens
        self.packed_examples += other.packed_examples


@dataclass
class SftSourceStats:
    segments: int = 0
    segments_with_history: int = 0
    history_reduced: int = 0
    history_turns_dropped_total: int = 0
    segments_with_scratchpad: int = 0
    scratchpad_removed: int = 0
    segments_with_system_prompt: int = 0
    system_prompt_dropped: int = 0

    def add(self, other: SftSourceStats) -> None:
        self.segments += other.segments
        self.segments_with_history += other.segments_with_history
        self.history_reduced += other.history_reduced
        self.history_turns_dropped_total += other.history_turns_dropped_total
        self.segments_with_scratchpad += other.segments_with_scratchpad
        self.scratchpad_removed += other.scratchpad_removed
        self.segments_with_system_prompt += other.segments_with_system_prompt
        self.system_prompt_dropped += other.system_prompt_dropped


@dataclass
class TrainingViewAnalysis:
    overall: BucketStats = field(default_factory=BucketStats)
    by_source: dict[SourceInfo, BucketStats] = field(default_factory=dict)
    by_component: dict[str, BucketStats] = field(default_factory=dict)
    sft_by_source: dict[SourceInfo, SftSourceStats] = field(default_factory=dict)
    sft_context_stats: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _SegmentContribution:
    doc_id: str
    component_name: str | None
    token_count: int
    loss_token_count: int


class _ProgressReporter:
    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at

    def maybe_report(
        self,
        *,
        analysis: TrainingViewAnalysis,
        capacity_estimator: CompositeCapacityEstimator | None,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self.last_report_at < PROGRESS_LOG_EVERY_SECONDS:
            return

        total_elapsed = max(now - self.started_at, 1e-9)
        classified_segments = sum(
            stats.segments
            for (src_id, _), stats in analysis.by_source.items()
            if src_id != UNCLASSIFIED_VALUE
        )
        unclassified_segments = analysis.overall.segments - classified_segments
        console_log(
            "analyze",
            (
                f"progress: {analysis.overall.examples:_} examples, "
                f"{analysis.overall.segments:_} segments, "
                f"{analysis.overall.total_tokens:_} total_tokens, "
                f"{analysis.overall.loss_only_tokens:_} loss_only_tokens, "
                f"{analysis.overall.examples / total_elapsed:.1f} ex/s, "
                f"{analysis.overall.total_tokens / total_elapsed:.1f} tok/s, "
                f"{unclassified_segments:_} unclassified segments"
            ),
        )
        if capacity_estimator is not None:
            undecided = capacity_estimator.undecided_source_names()
            console_log(
                "analyze",
                f"undecided sources ({len(undecided)}): {', '.join(undecided) if undecided else '(none)'}",
            )
        self.last_report_at = now


def load_training_view_source_lookup(
    output_root: str,
    config: TrainingViewConfig,
) -> dict[str, SourceInfo]:
    source_lookup: dict[str, SourceInfo] = {}
    for snapshot_id in _iter_snapshot_ids(config):
        source_lookup.update(_load_snapshot_source_lookup(output_root, snapshot_id))
    return source_lookup


def scan_training_view(
    source: Iterable[TrainingExample],
    *,
    source_lookup: Mapping[str, SourceInfo],
    batch_size: int,
    max_training_units: int,
    capacity_estimator: CompositeCapacityEstimator | None = None,
) -> TrainingViewAnalysis:
    analysis = TrainingViewAnalysis()
    progress = _ProgressReporter()

    if capacity_estimator is not None:
        undecided = capacity_estimator.undecided_source_names()
        console_log(
            "analyze",
            f"undecided sources ({len(undecided)}): {', '.join(undecided) if undecided else '(none)'}",
        )

    for example in iter_training_budget_examples(
        source,
        batch_size=batch_size,
        max_training_units=max_training_units,
    ):
        analysis.overall.add(_example_stats(example))
        for contribution in _segment_contributions(example):
            source_info = resolve_source_info(contribution.doc_id, source_lookup)
            segment_stats = BucketStats(
                segments=1,
                total_tokens=contribution.token_count,
                loss_only_tokens=contribution.loss_token_count,
            )
            analysis.by_source.setdefault(source_info, BucketStats()).add(segment_stats)
            if contribution.component_name is not None:
                analysis.by_component.setdefault(
                    contribution.component_name,
                    BucketStats(),
                ).add(segment_stats)

        add_sft_source_diagnostics(
            example,
            source_lookup=source_lookup,
            by_source=analysis.sft_by_source,
        )
        progress.maybe_report(
            analysis=analysis,
            capacity_estimator=capacity_estimator,
        )

    if capacity_estimator is not None and analysis.overall.loss_only_tokens >= max_training_units:
        capacity_estimator.mark_budget_reached()

    progress.maybe_report(
        analysis=analysis,
        capacity_estimator=capacity_estimator,
        force=True,
    )
    analysis.sft_context_stats = sft_stats_to_dict(
        getattr(source, "last_iteration_stats", None)
    )
    return analysis


def iter_training_budget_examples(
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


def resolve_source_info(
    materialized_record_id: str,
    source_lookup: Mapping[str, SourceInfo],
) -> SourceInfo:
    candidate = materialized_record_id
    while True:
        source_info = source_lookup.get(candidate)
        if source_info is not None:
            return source_info
        if "." not in candidate:
            return (UNCLASSIFIED_VALUE, UNCLASSIFIED_VALUE)
        candidate = candidate.rsplit(".", 1)[0]


def add_sft_source_diagnostics(
    example: TrainingExample,
    *,
    source_lookup: Mapping[str, SourceInfo],
    by_source: dict[SourceInfo, SftSourceStats],
) -> None:
    raw_segments = example.metadata.get("segments", [])
    if not isinstance(raw_segments, list):
        return

    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            continue
        if str(raw_segment.get(SEGMENT_KIND, "")) != SEGMENT_KIND_PROMPT_COMPLETION:
            continue

        doc_id = str(raw_segment.get(SEGMENT_DOC_ID, example.uid))
        stats = by_source.setdefault(
            resolve_source_info(doc_id, source_lookup),
            SftSourceStats(),
        )
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


def sft_stats_to_dict(stats: Any) -> dict[str, int]:
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


def _iter_snapshot_ids(config: TrainingViewConfig) -> list[str]:
    if isinstance(config, SnapshotTrainingViewConfig):
        return [config.materialized_config.snapshot_id]
    if isinstance(config, CompositeTrainingViewConfig):
        result: list[str] = []
        seen: set[str] = set()
        for source in config.sources:
            for snapshot_id in _iter_snapshot_ids(source.view_config):
                if snapshot_id not in seen:
                    seen.add(snapshot_id)
                    result.append(snapshot_id)
        return result
    raise TypeError(f"Unsupported training view config type: {type(config)!r}")


def _load_snapshot_source_lookup(output_root: str, snapshot_id: str) -> dict[str, SourceInfo]:
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


def _input_ids(example: TrainingExample) -> list[int]:
    token_ids = example.payload.get("input_ids")
    if not isinstance(token_ids, list):
        raise RuntimeError(f"Training example '{example.uid}' is missing payload['input_ids'].")
    return [int(value) for value in token_ids]


def _loss_mask(example: TrainingExample) -> list[float]:
    loss_mask = example.payload.get("loss_mask")
    if not isinstance(loss_mask, list):
        raise RuntimeError(f"Training example '{example.uid}' is missing payload['loss_mask'].")
    return [float(value) for value in loss_mask]


def _segment_token_stats(segment: Mapping[str, Any]) -> tuple[int, int] | None:
    if SEGMENT_PROMPT_LEN in segment and SEGMENT_COMPLETION_LEN in segment:
        prompt_len = int(segment[SEGMENT_PROMPT_LEN])
        completion_len = int(segment[SEGMENT_COMPLETION_LEN])
        loss_len = completion_len - (1 if prompt_len == 0 and completion_len > 0 else 0)
        return (prompt_len + completion_len, loss_len)

    if (
        SEGMENT_WINDOW_START in segment
        and SEGMENT_WINDOW_END_EXCLUDED in segment
        and SEGMENT_WINDOW_NEW_TOK_START in segment
    ):
        start = int(segment[SEGMENT_WINDOW_START])
        end_excluded = int(segment[SEGMENT_WINDOW_END_EXCLUDED])
        new_tok_start = int(segment[SEGMENT_WINDOW_NEW_TOK_START])
        input_len = end_excluded - start
        overlap_len = 0 if start == 0 else max(0, min(new_tok_start, end_excluded) - start)
        zero_count = overlap_len if overlap_len > 0 else (1 if input_len > 0 else 0)
        return (input_len, max(0, input_len - zero_count))
    return None


def _example_stats(example: TrainingExample) -> BucketStats:
    segments = example.metadata.get("segments", [])
    segment_count = len(segments) if isinstance(segments, list) and segments else 1
    return BucketStats(
        examples=1,
        segments=segment_count,
        total_tokens=len(_input_ids(example)),
        loss_only_tokens=int(sum(_loss_mask(example))),
        packed_examples=1 if segment_count > 1 else 0,
    )


def _segment_contributions(example: TrainingExample) -> list[_SegmentContribution]:
    total_tokens = len(_input_ids(example))
    total_loss_tokens = int(sum(_loss_mask(example)))
    top_level_component = example.metadata.get(SEGMENT_COMPONENT_NAME)
    raw_segments = example.metadata.get("segments", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        return [
            _SegmentContribution(
                doc_id=example.uid,
                component_name=str(top_level_component) if top_level_component is not None else None,
                token_count=total_tokens,
                loss_token_count=total_loss_tokens,
            )
        ]

    contributions: list[_SegmentContribution] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            continue
        stats = _segment_token_stats(raw_segment)
        if stats is None:
            continue
        component_name = raw_segment.get(SEGMENT_COMPONENT_NAME, top_level_component)
        contributions.append(
            _SegmentContribution(
                doc_id=str(raw_segment.get(SEGMENT_DOC_ID, example.uid)),
                component_name=str(component_name) if component_name is not None else None,
                token_count=stats[0],
                loss_token_count=stats[1],
            )
        )

    if not contributions:
        return [
            _SegmentContribution(
                doc_id=example.uid,
                component_name=str(top_level_component) if top_level_component is not None else None,
                token_count=total_tokens,
                loss_token_count=total_loss_tokens,
            )
        ]

    token_remainder = total_tokens - sum(item.token_count for item in contributions)
    loss_remainder = total_loss_tokens - sum(item.loss_token_count for item in contributions)
    if token_remainder != 0 or loss_remainder != 0:
        first = contributions[0]
        contributions[0] = _SegmentContribution(
            doc_id=first.doc_id,
            component_name=first.component_name,
            token_count=first.token_count + token_remainder,
            loss_token_count=first.loss_token_count + loss_remainder,
        )
    return contributions
