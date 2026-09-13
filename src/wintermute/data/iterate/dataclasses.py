from __future__ import annotations

from abc import ABC
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List

from wintermute.data.snapshot.materialized_iterable import MaterializedConfig
from wintermute.data.transform import RecordTransformConfig


class TrainingViewConfig(ABC):
    @property
    def display_name(self) -> str:
        raise NotImplementedError(f"Unsupported training view config type: {type(self)!r}")

    def to_dict(self) -> Dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError(f"Unsupported training view config type: {type(self)!r}")
        return asdict(self)


@dataclass(frozen=True)
class SnapshotTrainingViewConfig(TrainingViewConfig):
    materialized_config: MaterializedConfig
    include_prompt_in_loss: bool = False
    record_transform: RecordTransformConfig | None = None

    @property
    def display_name(self) -> str:
        transform = self.record_transform
        if transform is None:
            return self.materialized_config.snapshot_id
        return f"{self.materialized_config.snapshot_id}.{transform.kind}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "snapshot",
            "snapshot_id": self.materialized_config.snapshot_id,
            "split": self.materialized_config.split,
            "shard_offset_ratio": self.materialized_config.shard_offset_ratio,
            "strip_newlines_around_special_tokens": (
                self.materialized_config.strip_newlines_around_special_tokens
            ),
            "include_prompt_in_loss": self.include_prompt_in_loss,
            "record_transform": (
                self.record_transform.to_dict()
                if self.record_transform is not None
                else None
            ),
        }


@dataclass(frozen=True)
class CompositeTrainingSource:
    weight: float
    view_config: TrainingViewConfig
    max_restarts: int = 0

    @property
    def display_name(self) -> str:
        return self.view_config.display_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight": self.weight,
            "view_config": self.view_config.to_dict(),
            "max_restarts": self.max_restarts,
        }


@dataclass(frozen=True)
class CompositeTrainingViewConfig(TrainingViewConfig):
    sources: List[CompositeTrainingSource]
    seed: int = 1234
    mix_unit: str = "loss_tokens"

    @property
    def display_name(self) -> str:
        source_names = ", ".join(source.display_name for source in self.sources)
        return f"composite[{source_names}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "composite",
            "sources": [source.to_dict() for source in self.sources],
            "seed": self.seed,
            "mix_unit": self.mix_unit,
        }


@dataclass(frozen=True)
class WindowingPolicy:
    window_max_len: int = 4096
    overlap: int = 512
    min_new_tokens: int = 1024
    align_last_window_to_end: bool = True
    add_eos_at_end_of_doc: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> WindowingPolicy:
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)      


@dataclass(frozen=True)
class PackingPolicy:
    enabled: bool = True
    max_docs_per_pack: int = 3
    target_fill_ratio: float = 0.80
    example_buffer_size: int = 200

    @classmethod
    def from_dict(cls, data: dict) -> PackingPolicy:
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)   


@dataclass(frozen=True)
class WindowDefinition:
    start: int
    end_excluded: int  
    new_tok_start: int  

    def length(self) -> int:
        return self.end_excluded - self.start
    
    def new_tok_length(self) -> int:
        return self.end_excluded - self.new_tok_start    
    
    def overlap_length(self) -> int:
        if self.start == 0:
            return 0
        return max(0, min(self.new_tok_start, self.end_excluded) - self.start)    

@dataclass()
class TrainingExample:
    uid: str
    payload: Dict[str, Any]
    work_units: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def effective_units(self) -> int:
        return int(self.work_units)
