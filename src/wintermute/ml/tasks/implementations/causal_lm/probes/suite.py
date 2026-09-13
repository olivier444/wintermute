from __future__ import annotations

from dataclasses import dataclass, replace

from wintermute.ml.tasks.implementations.causal_lm.probes.base import ProbeSampling, ProbeSuite
from wintermute.ml.tasks.implementations.causal_lm.probes.config import ProbeConfig
from wintermute.ml.tasks.implementations.causal_lm.probes.metrics import ProbeMetric


@dataclass(frozen=True)
class ProbeSamplingGroup:
    sampling: ProbeSampling
    metrics: tuple[ProbeMetric, ...]
    observable_sample_count: int = 0


@dataclass(frozen=True)
class ProbeExecutionPlan:
    suite: ProbeSuite
    sampling_groups: tuple[ProbeSamplingGroup, ...]


@dataclass
class _SamplingGroupBuilder:
    sampling: ProbeSampling
    metrics: list[ProbeMetric]
    observable_sample_count: int = 0

    def include_sampling(self, sampling: ProbeSampling) -> None:
        if sampling.sample_count > self.sampling.sample_count:
            self.sampling = replace(
                self.sampling,
                sample_count=sampling.sample_count,
            )


@dataclass
class RuntimeProbeSuite:
    config: ProbeConfig
    suite: ProbeSuite
    _macro_steps_since_probe: int = 0

    def is_probe_due(self) -> bool:
        self._macro_steps_since_probe += 1
        if self._macro_steps_since_probe < self.config.probe_modulus:
            return False

        self._macro_steps_since_probe = 0
        return True

    def execution_plan(self) -> ProbeExecutionPlan | None:
        selected_names = set(self.config.tensorboard_metrics)
        selected_metrics = tuple(
            metric
            for metric in self.suite.metrics
            if metric.name in selected_names
        )
        groups: dict[tuple[object, ...], _SamplingGroupBuilder] = {}

        for metric in selected_metrics:
            key = metric.sampling.distribution_key()
            group = groups.get(key)
            if group is None:
                groups[key] = _SamplingGroupBuilder(metric.sampling, [metric])
                continue
            group.include_sampling(metric.sampling)
            group.metrics.append(metric)

        if self.config.slack_observable:
            observable_sampling = self.suite.observable_sampling
            key = observable_sampling.distribution_key()
            group = groups.get(key)
            if group is None:
                groups[key] = _SamplingGroupBuilder(
                    observable_sampling,
                    [],
                    observable_sample_count=observable_sampling.sample_count,
                )
            else:
                group.include_sampling(observable_sampling)
                group.observable_sample_count = observable_sampling.sample_count

        if not groups:
            return None

        return ProbeExecutionPlan(
            suite=self.suite,
            sampling_groups=tuple(
                ProbeSamplingGroup(
                    sampling=group.sampling,
                    metrics=tuple(group.metrics),
                    observable_sample_count=group.observable_sample_count,
                )
                for group in groups.values()
            ),
        )
