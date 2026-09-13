from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProbeConfig:
    preset: str
    probe_modulus: int = 1
    tensorboard_metrics: tuple[str, ...] = ()
    slack_observable: bool = False

    def __post_init__(self) -> None:
        if not self.preset.strip():
            raise ValueError("ProbeConfig.preset must be a non-empty string")
        if self.probe_modulus <= 0:
            raise ValueError("ProbeConfig.probe_modulus must be positive")
        if any(not metric.strip() for metric in self.tensorboard_metrics):
            raise ValueError("ProbeConfig.tensorboard_metrics must contain non-empty strings")
        if len(set(self.tensorboard_metrics)) != len(self.tensorboard_metrics):
            raise ValueError("ProbeConfig.tensorboard_metrics must be unique")

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        context: str = "probe config",
    ) -> ProbeConfig:
        preset = data.get("preset")
        if not isinstance(preset, str):
            raise TypeError(f"{context}.preset must be a string")

        probe_modulus = data.get("probe_modulus", 1)
        if type(probe_modulus) is not int:
            raise TypeError(f"{context}.probe_modulus must be an integer")

        raw_metrics = data.get("tensorboard_metrics", [])
        if not isinstance(raw_metrics, list):
            raise TypeError(f"{context}.tensorboard_metrics must be a list")
        if any(not isinstance(metric, str) for metric in raw_metrics):
            raise TypeError(f"{context}.tensorboard_metrics must contain strings")

        slack_observable = data.get("slack_observable", False)
        if not isinstance(slack_observable, bool):
            raise TypeError(f"{context}.slack_observable must be a boolean")

        return cls(
            preset=preset,
            probe_modulus=probe_modulus,
            tensorboard_metrics=tuple(raw_metrics),
            slack_observable=slack_observable,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tensorboard_metrics"] = list(self.tensorboard_metrics)
        return data
