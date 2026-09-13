from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Mapping, Optional
from wintermute.ml.models.specifications import ModelSpecification
from wintermute.data.iterate.dataclasses import TrainingViewConfig
from wintermute.ops.presets.registry import get_training_view_preset
from wintermute.tools.files import yaml_load, yaml_save
from wintermute.tools.params import (
    get_bool,
    get_int,
    get_optional_int,
    get_optional_list,
    get_optional_mapping,
    get_optional_positive_int,
    get_optional_str,
    get_str,
)
from wintermute.tools.slack.config import SlackConfig
from wintermute.ml.training.task_specification import TaskSpecification


@dataclass
class TrainingConfig:
    task_config: TaskSpecification
    model_config: ModelSpecification
    reference_run_id: Optional[str]
    training_view_preset: str
    training_view_config: TrainingViewConfig
    lr_config: LrConfig
    optimizer_config: OptimizerConfig
    trainer_config: TrainerConfig
    evaluation_configs: List[EvaluationConfig]
    benchmark_config: Optional[BenchmarkConfig] = None
    reference_checkpoint: Optional[str] = None
    load_reference_optimizer_state: bool = False
    baseline_config: Dict[str, Any] = field(default_factory=dict)
    slack_config: Optional[SlackConfig] = None
    slack_config_payload: Optional[Dict[str, Any]] = field(default=None, repr=False)
    probe_configs: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        load_slack_config: bool = True,
    ) -> TrainingConfig:
        d = dict(data)
        if "task_config" not in d or d["task_config"] is None:
            raise ValueError("training config must define 'task_config'")
        d["task_config"] = TaskSpecification.from_dict(d["task_config"])

        d["model_config"] = ModelSpecification.from_dict(d["model_config"])
        d["reference_checkpoint"] = get_optional_str(d, "reference_checkpoint")
        d["load_reference_optimizer_state"] = get_bool(d, "load_reference_optimizer_state", False)
        if d["reference_checkpoint"] is not None and not d["reference_checkpoint"].strip():
            raise ValueError("training config 'reference_checkpoint' must be a non-empty selector when provided.")
        if d["reference_checkpoint"] is not None and not d.get("reference_run_id"):
            raise ValueError("training config 'reference_checkpoint' requires 'reference_run_id'.")
        if d["load_reference_optimizer_state"] and d["reference_checkpoint"] is None:
            raise ValueError(
                "training config 'load_reference_optimizer_state' requires 'reference_checkpoint'."
            )
        if "training_view_config" in d:
            raise ValueError(
                "training config no longer supports inline 'training_view_config'; "
                "use 'training_view_preset'"
            )
        training_view_preset = get_str(d, "training_view_preset", "").strip()
        if not training_view_preset:
            raise ValueError("training config must define 'training_view_preset'")
        resolved_training_view = get_training_view_preset(training_view_preset)
        d["training_view_preset"] = resolved_training_view.name
        d["training_view_config"] = resolved_training_view.config
        d["lr_config"] = LrConfig.from_dict(d["lr_config"])
        d["optimizer_config"] = OptimizerConfig.from_dict(d["optimizer_config"])
        raw_trainer_config = get_optional_mapping(d, "trainer_config")
        if raw_trainer_config is None:
            raise ValueError("training config must define 'trainer_config'")
        d["trainer_config"] = TrainerConfig.from_dict(dict(raw_trainer_config))

        slack_cfg = get_optional_mapping(d, "slack_config")
        if load_slack_config:
            d["slack_config"] = SlackConfig.from_dict(dict(slack_cfg)) if slack_cfg is not None else None
        else:
            d["slack_config"] = None
            d["slack_config_payload"] = dict(slack_cfg) if slack_cfg is not None else None

        baseline_config = get_optional_mapping(d, "baseline_config")
        d["baseline_config"] = dict(baseline_config) if baseline_config is not None else {}
        d["probe_configs"] = _load_probe_configs(get_optional_list(d, "probe_configs"))
        d["evaluation_configs"] = _load_evaluation_configs(
            get_optional_list(d, "evaluation_configs"),
            default_task=d["task_config"],
        )
        raw_benchmark_config = get_optional_mapping(d, "benchmark_config")
        d["benchmark_config"] = (
            BenchmarkConfig.from_dict(raw_benchmark_config)
            if raw_benchmark_config is not None
            else None
        )
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("training_view_config", None)
        skipped_slack_config = data.pop("slack_config_payload", None)
        data["lr_config"] = self.lr_config.to_dict()
        data["trainer_config"] = self.trainer_config.to_dict()
        data["evaluation_configs"] = [spec.to_dict() for spec in self.evaluation_configs]
        data["benchmark_config"] = (
            self.benchmark_config.to_dict()
            if self.benchmark_config is not None
            else None
        )
        data["slack_config"] = (
            self.slack_config.to_dict()
            if self.slack_config is not None
            else skipped_slack_config
        )
        data["probe_configs"] = [dict(config) for config in self.probe_configs]
        return data

    def save(self, directory: str, file_name: str) -> None:
        yaml_save(self, directory, file_name)

    @classmethod
    def load(cls, directory: str, file_name: str) -> TrainingConfig:
        return cls.from_dict(yaml_load(directory, file_name))

    @property
    def primary_evaluation_config(self) -> EvaluationConfig:
        return next(spec for spec in self.evaluation_configs if spec.is_primary)


@dataclass
class TrainerConfig:
    checkpoint_every_batch: int
    max_training_units: int
    batch_size: int
    grad_accum_size_units: int
    seed: int
    use_bf16: bool
    model_save_pcts: List[float] = field(default_factory=list)
    gradient_noise_update_batch_count: int = 4
    gradient_noise_modulus: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "TrainerConfig":
        d = dict(data)
        if "max_epochs" in d:
            raise ValueError("trainer config no longer supports 'max_epochs'.")
        d["max_training_units"] = get_int(d, "max_training_units", 0)
        if d["max_training_units"] <= 0:
            raise ValueError("trainer config 'max_training_units' must be positive.")
        d["gradient_noise_update_batch_count"] = get_int(d, "gradient_noise_update_batch_count", 0)
        d["gradient_noise_modulus"] = max(1, get_int(d, "gradient_noise_modulus", 1))
        d["model_save_pcts"] = _load_model_save_pcts(get_optional_list(d, "model_save_pcts"))
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerConfig:
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.05
    grad_clip: float = 1

    @classmethod
    def from_dict(cls, data: dict) -> OptimizerConfig:
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LrConfig:
    end_lr: float
    max_lr: float
    warmup_length_ratio: float
    cos_decay_start_ratio: float
    start_lr: float = 0

    @classmethod
    def from_dict(cls, data: dict) -> LrConfig:
        if "min_lr" in data:
            raise ValueError("lr config uses 'end_lr'; 'min_lr' is unsupported.")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_lr": self.start_lr,
            "max_lr": self.max_lr,
            "end_lr": self.end_lr,
            "warmup_length_ratio": self.warmup_length_ratio,
            "cos_decay_start_ratio": self.cos_decay_start_ratio,
        }


@dataclass
class BenchmarkConfig:
    every_n_checkpoints: int
    values: Dict[str, Any]

    @property
    def tasks(self) -> List[str]:
        return list(self.values["tasks"])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkConfig":
        payload = dict(data)
        every_n_checkpoints = get_int(payload, "every_n_checkpoints", 0)
        if every_n_checkpoints <= 0:
            raise ValueError("benchmark_config 'every_n_checkpoints' must be positive")
        payload.pop("every_n_checkpoints", None)

        tasks = get_optional_list(payload, "tasks")
        if not tasks:
            raise ValueError("benchmark_config 'tasks' must be a non-empty list")
        if any(not isinstance(task, str) or not task.strip() for task in tasks):
            raise TypeError("benchmark_config 'tasks' must contain non-empty task names")
        payload["tasks"] = [task.strip() for task in tasks]

        if "include_path" in payload:
            raise ValueError("benchmark_config does not support custom task include_path")
        if "device" in payload:
            raise ValueError("benchmark_config uses the training device and must not define 'device'")

        return cls(every_n_checkpoints=every_n_checkpoints, values=payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "every_n_checkpoints": self.every_n_checkpoints,
            **self.values,
        }


@dataclass
class EvaluationConfig:
    name: str
    view_preset: str
    view_config: TrainingViewConfig
    task_config: TaskSpecification
    batch_size: Optional[int] = None
    max_eval_units: Optional[int] = None
    is_primary: bool = False
    evaluation_modulus: int = 1

    def __post_init__(self) -> None:
        self.evaluation_modulus = max(1, int(self.evaluation_modulus))
        if self.is_primary and self.evaluation_modulus != 1:
            raise ValueError("The primary evaluation config must use evaluation_modulus=1.")

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        default_task: TaskSpecification,
    ) -> "EvaluationConfig":
        payload = dict(data)
        name = get_str(payload, "name", "default") or "default"
        task_payload = get_optional_mapping(payload, "task_config")
        if "view_config" in payload:
            raise ValueError(
                f"Evaluation config '{name}' no longer supports inline 'view_config'; "
                "use 'view_preset'"
            )
        view_preset = get_str(payload, "view_preset", "").strip()
        if not view_preset:
            raise ValueError(f"Evaluation config '{name}' must define 'view_preset'.")
        resolved_view = get_training_view_preset(view_preset)

        return cls(
            name=name,
            view_preset=resolved_view.name,
            view_config=resolved_view.config,
            task_config=(
                TaskSpecification.from_dict(dict(task_payload))
                if task_payload is not None
                else default_task
            ),
            batch_size=get_optional_positive_int(
                payload,
                "batch_size",
                field_name=f"evaluation config '{name}' batch_size",
            ),
            max_eval_units=_load_max_eval_units(payload),
            is_primary=get_bool(payload, "is_primary", False),
            evaluation_modulus=get_int(payload, "evaluation_modulus", 1),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("view_config", None)
        return data


def _load_evaluation_configs(
    raw_specs: list[Any] | None,
    *,
    default_task: TaskSpecification,
) -> List[EvaluationConfig]:
    if not raw_specs:
        raise ValueError("'evaluation_configs' must be a non-empty list.")

    specs = [EvaluationConfig.from_dict(spec, default_task=default_task) for spec in raw_specs]
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Duplicate evaluation config name detected.")
    if sum(1 for spec in specs if spec.is_primary) != 1:
        raise ValueError("Exactly one evaluation config must declare 'is_primary=true'.")
    return specs


def _load_probe_configs(raw_configs: list[Any] | None) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for index, raw_config in enumerate(raw_configs or []):
        context = f"probe_configs[{index}]"
        if not isinstance(raw_config, Mapping):
            raise TypeError(f"{context} must be a mapping")
        configs.append(dict(raw_config))
    return configs


def _load_max_eval_units(payload: dict) -> Optional[int]:
    max_eval_units = get_optional_int(payload, "max_eval_units")
    if max_eval_units is None or max_eval_units <= 0:
        return None
    return max_eval_units


def _load_model_save_pcts(raw_pcts: list[Any] | None) -> List[float]:
    if raw_pcts is None:
        return []

    pcts = sorted({float(pct) for pct in raw_pcts})
    if any(pct <= 0.0 or pct >= 100.0 for pct in pcts):
        raise ValueError("training config 'model_save_pcts' values must be strictly between 0 and 100.")
    return pcts
