from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from wintermute.tools.files import yaml_load
from wintermute.tools.params import get_float, get_optional_list, get_optional_mapping, get_optional_str
from wintermute.tools.slack.config import SlackConfig


def _normalize_inference_device(value: str | None) -> str:
    clean = (value or "cpu").strip().lower()
    if not clean or clean == "cpu":
        return "cpu"
    if clean in {"gpu", "cuda"}:
        return "cuda"
    if clean.startswith("cuda:"):
        return clean
    raise ValueError("inference endpoint device must be 'cpu' or 'gpu'")


@dataclass(frozen=True)
class InferenceEndpointModel:
    name: str
    run_id: str
    external_model_name: str | None = None


@dataclass(frozen=True)
class InferenceEndpointConfig:
    slack_config: SlackConfig
    models: tuple[InferenceEndpointModel, ...]
    device: str = "cpu"
    system_prompt: str = ""
    poll_interval_sec: float = 0.25

    @property
    def default_model_name(self) -> str:
        return self.models[0].name

    @property
    def model_targets(self) -> dict[str, tuple[str, str | None]]:
        return {
            model.name: (model.run_id, model.external_model_name)
            for model in self.models
        }

    @classmethod
    def load(cls, path: str | Path) -> "InferenceEndpointConfig":
        config_path = Path(path)
        payload = yaml_load(config_path.parent.as_posix(), config_path.name)
        if not isinstance(payload, dict):
            raise TypeError("inference endpoint config root must be a YAML mapping")

        wrapped = get_optional_mapping(payload, "configuration")
        data = dict(wrapped) if wrapped is not None else dict(payload)

        slack_payload = get_optional_mapping(data, "slack_config")
        if slack_payload is None:
            raise ValueError("inference endpoint config requires 'slack_config'")

        poll_interval_sec = get_float(data, "poll_interval_sec", 0.25)
        if poll_interval_sec <= 0:
            raise ValueError("inference endpoint poll_interval_sec must be > 0")

        return cls(
            slack_config=SlackConfig.from_dict(dict(slack_payload)),
            models=_load_models(data),
            device=_normalize_inference_device(get_optional_str(data, "device")),
            system_prompt=(get_optional_str(data, "system_prompt") or "").strip(),
            poll_interval_sec=poll_interval_sec,
        )


def _load_models(data: Mapping[str, Any]) -> tuple[InferenceEndpointModel, ...]:
    raw_models = get_optional_list(data, "models")
    if raw_models is None or len(raw_models) == 0:
        raise ValueError("inference endpoint config requires a non-empty 'models' list")

    models = tuple(_parse_model(raw_model, index=index) for index, raw_model in enumerate(raw_models))

    seen_names: set[str] = set()
    for model in models:
        key = model.name.lower()
        if key in seen_names:
            raise ValueError(f"duplicate inference model name '{model.name}'")
        seen_names.add(key)

    return models


def _parse_model(raw_model: Any, *, index: int) -> InferenceEndpointModel:
    context = f"models[{index}]"
    if not isinstance(raw_model, Mapping):
        raise TypeError(f"{context} must be an object")

    data = dict(raw_model)
    name = (get_optional_str(data, "name") or "").strip()
    if not name:
        raise ValueError(f"{context} requires 'name'")

    run_id = (get_optional_str(data, "run_id") or "").strip() or None
    task_run_id = (get_optional_str(data, "task_run_id") or "").strip() or None
    external_model_name = (get_optional_str(data, "external_model_name") or "").strip() or None

    if external_model_name is not None:
        if run_id is not None:
            raise ValueError(f"{context} uses 'external_model_name'; use 'task_run_id' instead of 'run_id'")
        if task_run_id is None:
            raise ValueError(f"{context} requires 'task_run_id' with 'external_model_name'")
        return InferenceEndpointModel(
            name=name,
            run_id=task_run_id,
            external_model_name=external_model_name,
        )

    if task_run_id is not None:
        raise ValueError(f"{context} uses 'task_run_id' without 'external_model_name'")
    if run_id is None:
        raise ValueError(f"{context} requires 'run_id'")

    return InferenceEndpointModel(name=name, run_id=run_id)
