#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from wintermute.data.transform import RecordTransformConfig
from wintermute.tools.files import get_index_dir


@dataclass(frozen=True)
class DedupConfig:
    reference_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SnapshotConfig:
    snapshot_id: str                                # uid for this config
    output_root: str
    split_sizes_chars: Dict[str, int]               # eg: { 'train': 4e8, 'valid': 3e7 }
    dedup: DedupConfig | None
    excluded_languages: List[str]
    seed: int
    source_configs: Dict[str, SnapshotSourceConfig] # indexed by source.src_id
    max_bytes: int = 100 * 1024 * 1024

    def get_index_path(self) -> str:
        return get_index_dir(self.output_root)
    

@dataclass(frozen=True)
class SnapshotSourceConfig:
    weight: float
    min_record_size_char: int
    max_record_size_char: int
    record_transform: RecordTransformConfig | None = None
    oversampling: int = 1
    excluded_snapshot_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.oversampling < 1:
            raise ValueError(f"SnapshotSourceConfig.oversampling must be >= 1, got {self.oversampling}")
        object.__setattr__(
            self,
            "excluded_snapshot_ids",
            [str(snapshot_id).strip() for snapshot_id in self.excluded_snapshot_ids if str(snapshot_id).strip()],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SnapshotSourceConfig:
        data = dict(payload)
        raw_transform = data.get("record_transform")
        if raw_transform is not None:
            data["record_transform"] = RecordTransformConfig.from_dict(raw_transform)

        return cls(**data)
