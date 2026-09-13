from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, List

import torch.nn as nn

from wintermute.ml.tasks.implementations.causal_lm.probes.base import (
    Probe,
    ProbeResult,
)
from wintermute.ml.tasks.implementations.causal_lm.probes.config import ProbeConfig
from wintermute.ml.tasks.implementations.causal_lm.probes.evaluation import (
    aggregate_probe_metrics_by_group,
)
from wintermute.ml.tasks.implementations.causal_lm.probes.executor import (
    CausalLmProbeExecutor,
)
from wintermute.ml.tasks.implementations.causal_lm.probes.presets import (
    get_probe_preset,
)
from wintermute.ml.tasks.implementations.causal_lm.probes.suite import (
    RuntimeProbeSuite,
)
from wintermute.ml.training.logger import Logger
from wintermute.ml.training.monitor import TrainingMonitor
from wintermute.tools.model import preserving_torch_rng_state


def _format_probe_observable(probe: Probe, result: ProbeResult) -> str:
    lines = [f"prompt: [{probe.prompt}]"]
    multiple_candidates = len(result.candidates) > 1

    for index, candidate in enumerate(result.candidates, start=1):
        label = f"candidate {index}" if multiple_candidates else "completion"
        completion = candidate.completion
        lines.append(f"{label}:")
        if completion.task is not None:
            lines.append(f"  task: [{completion.task}]")
        if completion.verbosity is not None:
            lines.append(f"  verbosity: [{completion.verbosity}]")
        if completion.format is not None:
            lines.append(f"  format: [{completion.format}]")
        if completion.scratchpad is not None:
            lines.append(f"  scratchpad: [{completion.scratchpad}]")
        lines.append(f"  final: [{completion.final}]")
        if candidate.score is not None:
            lines.append(f"  score: {candidate.score:.6g}")

    if result.metrics:
        rendered_metrics = ", ".join(
            f"{name}={value:.6g}"
            for name, value in sorted(result.metrics.items())
        )
        lines.append(f"metrics: {rendered_metrics}")

    return "\n".join(lines)


class CausalLmProbeMonitor(TrainingMonitor):
    def __init__(
        self,
        *,
        executor: CausalLmProbeExecutor,
        model: nn.Module,
        logger: Logger,
        suites: Iterable[RuntimeProbeSuite],
    ) -> None:
        self.executor = executor
        self.model = model
        self.logger = logger
        self.suites = list(suites)
        self._previews: dict[str, List[str]] = {}
        self._due_suites: list[RuntimeProbeSuite] = []

    @classmethod
    def from_configs(
        cls,
        *,
        executor: CausalLmProbeExecutor,
        model: nn.Module,
        logger: Logger,
        configs: Iterable[Mapping[str, Any]],
    ) -> "CausalLmProbeMonitor":
        suites = []
        for index, raw_config in enumerate(configs):
            config = ProbeConfig.from_dict(
                raw_config,
                context=f"probe_configs[{index}]",
            )
            suites.append(
                RuntimeProbeSuite(
                    config=config,
                    suite=get_probe_preset(config.preset),
                )
            )

        return cls(
            executor=executor,
            model=model,
            logger=logger,
            suites=suites,
        )

    @property
    def previews(self) -> Mapping[str, List[str]]:
        return self._previews

    def is_due(self) -> bool:
        self._due_suites = self.select_due_suites()
        return bool(self._due_suites)

    def run_due(self, *, global_step: int) -> None:
        due_suites = self._due_suites
        self._due_suites = []
        self.run(due_suites, global_step=global_step)

    def run_initial(self, *, global_step: int) -> None:
        self.run_all(global_step=global_step)

    def select_due_suites(self) -> list[RuntimeProbeSuite]:
        return [suite for suite in self.suites if suite.is_probe_due()]

    def run_all(self, *, global_step: int) -> None:
        self.run(self.suites, global_step=global_step)

    def run(
        self,
        suites: Iterable[RuntimeProbeSuite],
        *,
        global_step: int,
    ) -> None:
        with preserving_torch_rng_state():
            for runtime_suite in suites:
                plan = runtime_suite.execution_plan()
                if plan is None:
                    continue

                suite = plan.suite
                report = self.executor.run(self.model, plan)
                selected_metric_names = set(
                    runtime_suite.config.tensorboard_metrics
                )
                for metric_name in runtime_suite.config.tensorboard_metrics:
                    if metric_name not in report.metrics:
                        continue
                    self.logger.log_scalar(
                        f"{metric_name}/probes/{suite.name}",
                        report.metrics[metric_name],
                        global_step,
                    )
                grouped_metrics = aggregate_probe_metrics_by_group(report.results)
                for group, metrics in grouped_metrics.items():
                    for metric_name, value in metrics.items():
                        if metric_name not in selected_metric_names:
                            continue
                        self.logger.log_scalar(
                            f"{metric_name}/probes/{suite.name}/{group}",
                            value,
                            global_step,
                        )

                if not runtime_suite.config.slack_observable:
                    continue

                probes_by_id = {probe.id: probe for probe in suite.probes}
                for result in report.results:
                    preview_name = f"probes/{suite.name}/{result.probe_id}"
                    self._previews[preview_name] = [
                        _format_probe_observable(
                            probes_by_id[result.probe_id],
                            result,
                        )
                    ]
