from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


@dataclass(frozen=True)
class LmEvalConfig:
    path: Path | None
    values: Mapping[str, Any]
    device: torch.device
    sample_rate: float

    @classmethod
    def load(cls, path: str | Path) -> "LmEvalConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"lm-eval config not found: {config_path}")
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid lm-eval YAML config at {config_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"lm-eval YAML root must be a mapping: {config_path}")
        return cls.from_mapping(payload, path=config_path)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        path: Path | None = None,
        default_device: torch.device | None = None,
    ) -> "LmEvalConfig":
        payload = dict(values)
        if not payload.get("tasks"):
            raise ValueError("lm-eval config must define at least one task")
        return cls(
            path=path,
            values=payload,
            device=_resolve_device(payload.get("device"), default=default_device),
            sample_rate=_parse_sample_rate(payload.get("sample_rate", 0.0)),
        )

    @property
    def harness_values(self) -> dict[str, Any]:
        values = dict(self.values)
        values.pop("sample_rate", None)
        if include_path := values.get("include_path"):
            if self.path is None:
                raise ValueError(
                    "inline lm-eval config does not support relative include_path"
                )
            values["include_path"] = _resolve_include_path(include_path, self.path.parent)
        return values


def _resolve_device(
    raw_device: Any,
    *,
    default: torch.device | None = None,
) -> torch.device:
    if raw_device is None:
        return default or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean = str(raw_device).strip().lower()
    if clean == "gpu":
        clean = "cuda"
    device = torch.device(clean)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested by lm-eval config, but no GPU is available")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("lm-eval device must be 'cpu' or 'cuda'")
    return device


def _parse_sample_rate(raw_sample_rate: Any) -> float:
    if isinstance(raw_sample_rate, bool):
        raise TypeError("lm-eval sample_rate must be a number between zero and one")
    try:
        sample_rate = float(raw_sample_rate)
    except (TypeError, ValueError) as exc:
        raise TypeError("lm-eval sample_rate must be a number between zero and one") from exc
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("lm-eval sample_rate must be between zero and one")
    return sample_rate


def _resolve_include_path(raw_include_path: Any, config_dir: Path) -> Any:
    def resolve(path: Any) -> str:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = config_dir / candidate
        return candidate.resolve().as_posix()

    if isinstance(raw_include_path, (list, tuple)):
        return [resolve(path) for path in raw_include_path]
    return resolve(raw_include_path)
