from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tabulate import tabulate

from wintermute.data.iterate.segment_metadata import SEGMENT_SYSTEM_PROMPT_DROPPED
from wintermute.ml.tasks.implementations.causal_lm.analysis import (
    BucketStats,
    CompositeCapacityEstimator,
    SftSourceStats,
    TrainingViewAnalysis,
)


@dataclass(frozen=True)
class TrainingViewAnalysisReport:
    ascii_report: str
    overview_rows: list[dict[str, Any]]
    source_rows: list[dict[str, Any]]
    component_rows: list[dict[str, Any]] | None
    sft_context_overview_rows: list[dict[str, Any]] | None
    sft_context_source_rows: list[dict[str, Any]] | None


def build_training_view_analysis_report(
    analysis: TrainingViewAnalysis,
    *,
    config_name: str,
    is_composite: bool,
    top_sources: int,
    max_training_units: int,
    capacity_estimator: CompositeCapacityEstimator | None,
) -> TrainingViewAnalysisReport:
    limit = None if top_sources < 0 else top_sources
    overview_rows = [_overview_row(analysis.overall, len(analysis.by_source))]

    sources = sorted(
        analysis.by_source.items(),
        key=lambda item: (-item[1].loss_only_tokens, -item[1].examples, item[0][0]),
    )
    source_rows = _source_rows(
        sources if limit is None else sources[:limit],
        analysis.overall.loss_only_tokens,
    )

    component_rows = None
    if is_composite:
        components = sorted(
            analysis.by_component.items(),
            key=lambda item: (-item[1].loss_only_tokens, -item[1].examples, item[0]),
        )
        component_rows = _component_rows(
            components if limit is None else components[:limit],
            analysis.overall.loss_only_tokens,
        )

    sft_overview_rows = None
    sft_source_rows = None
    if analysis.sft_context_stats.get("sft_candidates", 0) > 0:
        sft_overview_rows = [_sft_overview_row(analysis.sft_context_stats)]
        sft_sources = sorted(
            analysis.sft_by_source.items(),
            key=lambda item: (
                -item[1].history_turns_dropped_total,
                -item[1].history_reduced,
                -item[1].scratchpad_removed,
                -item[1].system_prompt_dropped,
                -item[1].segments,
                item[0][0],
            ),
        )
        sft_source_rows = [
            _sft_source_row(source, stats)
            for source, stats in (sft_sources if limit is None else sft_sources[:limit])
        ]

    capacity_rows = None
    if capacity_estimator is not None:
        capacity_rows = [
            {**asdict(source), "required_training_units": max_training_units}
            for source in capacity_estimator.sources
        ]

    ascii_report = _ascii_report(
        config_name=config_name,
        top_sources=top_sources,
        overview_rows=overview_rows,
        source_rows=source_rows,
        component_rows=component_rows,
        capacity_rows=capacity_rows,
        capacity_result=capacity_estimator.result if capacity_estimator else None,
    )
    return TrainingViewAnalysisReport(
        ascii_report=ascii_report,
        overview_rows=overview_rows,
        source_rows=source_rows,
        component_rows=component_rows,
        sft_context_overview_rows=sft_overview_rows,
        sft_context_source_rows=sft_source_rows,
    )


def _overview_row(stats: BucketStats, source_count: int) -> dict[str, Any]:
    return {
        "scope": "ALL",
        **asdict(stats),
        "loss_token_ratio_pct": _pct(stats.loss_only_tokens, stats.total_tokens),
        "avg_total_tokens_per_example": _ratio(stats.total_tokens, stats.examples),
        "avg_loss_only_tokens_per_example": _ratio(stats.loss_only_tokens, stats.examples),
        "avg_segments_per_example": _ratio(stats.segments, stats.examples),
        "source_count": source_count,
    }


def _metric_values(stats: BucketStats, scope_loss_tokens: int) -> dict[str, Any]:
    return {
        "segments": stats.segments,
        "total_tokens": stats.total_tokens,
        "loss_only_tokens": stats.loss_only_tokens,
        "loss_token_share_pct": _pct(stats.loss_only_tokens, scope_loss_tokens),
        "loss_token_ratio_pct": _pct(stats.loss_only_tokens, stats.total_tokens),
        "avg_total_tokens_per_segment": _ratio(stats.total_tokens, stats.segments),
        "avg_loss_only_tokens_per_segment": _ratio(stats.loss_only_tokens, stats.segments),
    }


def _source_rows(
    rows: Sequence[tuple[tuple[str, str], BucketStats]],
    scope_loss_tokens: int,
) -> list[dict[str, Any]]:
    return [
        {"src_id": source[0], "src_label": source[1], **_metric_values(stats, scope_loss_tokens)}
        for source, stats in rows
    ]


def _component_rows(
    rows: Sequence[tuple[str, BucketStats]],
    scope_loss_tokens: int,
) -> list[dict[str, Any]]:
    return [
        {"component": component, **_metric_values(stats, scope_loss_tokens)}
        for component, stats in rows
    ]


def _sft_overview_row(stats: Mapping[str, int]) -> dict[str, Any]:
    candidates = int(stats.get("sft_candidates", 0))
    history_dropped = int(stats.get("sft_history_turns_dropped_total", 0))
    return {
        "scope": "ALL",
        **{key: int(value) for key, value in stats.items()},
        "rejected_pct": _pct(stats.get("sft_rejected", 0), candidates),
        "scratchpad_removed_pct": _pct(stats.get("sft_scratchpad_removed", 0), candidates),
        "scratchpad_removed_pct_of_scratchpad_examples": _pct(
            stats.get("sft_scratchpad_removed", 0),
            stats.get("sft_examples_with_scratchpad", 0),
        ),
        "history_reduced_pct": _pct(stats.get("sft_history_reduced", 0), candidates),
        "history_turns_dropped_avg": _ratio(
            history_dropped,
            stats.get("sft_examples_with_history", 0),
        ),
        "system_prompt_dropped_pct": _pct(
            stats.get("sft_system_prompt_dropped", 0), candidates
        ),
        "system_prompt_dropped_pct_of_system_prompt_examples": _pct(
            stats.get("sft_system_prompt_dropped", 0),
            stats.get("sft_examples_with_system_prompt", 0),
        ),
    }


def _sft_source_row(
    source: tuple[str, str],
    stats: SftSourceStats,
) -> dict[str, Any]:
    values = asdict(stats)
    values["sft_segments"] = values.pop("segments")
    return {
        "src_id": source[0],
        "src_label": source[1],
        **values,
        "history_reduced_pct_of_history_segments": _pct(
            stats.history_reduced, stats.segments_with_history
        ),
        "history_turns_dropped_avg": _ratio(
            stats.history_turns_dropped_total, stats.segments_with_history
        ),
        "scratchpad_removed_pct_of_scratchpad_segments": _pct(
            stats.scratchpad_removed, stats.segments_with_scratchpad
        ),
        SEGMENT_SYSTEM_PROMPT_DROPPED: stats.system_prompt_dropped,
        "system_prompt_dropped_pct_of_system_prompt_segments": _pct(
            stats.system_prompt_dropped, stats.segments_with_system_prompt
        ),
    }


def _ascii_report(
    *,
    config_name: str,
    top_sources: int,
    overview_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]] | None,
    capacity_rows: list[dict[str, Any]] | None,
    capacity_result: str | None,
) -> str:
    top = top_sources if top_sources > 0 else "all"
    sections = [("Overview", overview_rows)]
    if capacity_rows is not None:
        sections.append((f"Fast Capacity ({capacity_result})", capacity_rows))
    if component_rows is not None:
        sections.append(
            (f"Per Composite Component ({config_name}, top {top} by loss_only_tokens)", component_rows)
        )
    sections.append((f"Per Source ({config_name}, top {top} by loss_only_tokens)", source_rows))
    return "\n\n".join(f"=== {title} ===\n{_render_table(rows)}" for title, rows in sections) + "\n"


def _render_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    display_rows = [
        {key.replace("_", " "): value for key, value in row.items()}
        for row in rows
    ]
    return str(
        tabulate(
            display_rows,
            headers="keys",
            tablefmt="github",
            intfmt=",",
            floatfmt=",.2f",
        )
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _pct(numerator: int, denominator: int) -> float | None:
    ratio = _ratio(numerator, denominator)
    return 100.0 * ratio if ratio is not None else None
