from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from wintermute.ml.tasks.implementations.causal_lm.probes.metrics import ProbeMetric, compute_probe_metrics
from wintermute.ml.tasks.implementations.causal_lm.probes.base import Probe, ProbeCandidate, ProbeResult
from wintermute.ml.tokenization.chat_format import AssistantCompletion


def evaluate_probe(
    probe: Probe,
    completions: Iterable[AssistantCompletion],
    metrics: Sequence[ProbeMetric],
) -> ProbeResult:
    candidates = tuple(
        ProbeCandidate(
            completion=completion,
            score=(
                None
                if probe.verifier is None
                else float(probe.verifier.score(completion.final))
            ),
        )
        for completion in completions
    )
    scores = tuple(
        candidate.score
        for candidate in candidates
        if candidate.score is not None
    )
    return ProbeResult(
        probe_id=probe.id,
        candidates=candidates,
        metrics=compute_probe_metrics(metrics, scores),
    )


def aggregate_probe_metrics(results: Iterable[ProbeResult]) -> dict[str, float]:
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for name, value in result.metrics.items():
            values_by_metric[name].append(value)

    return {
        name: sum(values) / len(values)
        for name, values in values_by_metric.items()
        if values
    }


def aggregate_probe_metrics_by_group(
    results: Iterable[ProbeResult],
) -> dict[str, dict[str, float]]:
    results_by_group: dict[str, list[ProbeResult]] = defaultdict(list)
    for result in results:
        group, _, _ = result.probe_id.partition("/")
        results_by_group[group].append(result)

    return {
        group: aggregate_probe_metrics(group_results)
        for group, group_results in results_by_group.items()
    }
