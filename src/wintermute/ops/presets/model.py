from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from wintermute.data.get_raw.config import GetRawPreprocessorConfig
from wintermute.data.iterate.dataclasses import TrainingViewConfig
from wintermute.data.snapshot.config import DedupConfig, SnapshotSourceConfig
from wintermute.tools.files import get_raw_dir


@dataclass(frozen=True)
class DatasetPreset:
    uid: str
    name: str
    dataset_name: str
    text_fields: tuple[str, ...]
    get_raw_output_max_shards: int
    config_template: Optional[str] = None
    split: str = "train"
    revision: Optional[str] = None
    lan_field: Optional[str] = None
    lang: Optional[List[str]] = None
    seed: int = 1234
    get_raw_stat_fields: Optional[List[str]] = None
    get_raw_exclude_fields: Optional[List[str]] = None
    get_raw_data_files: Optional[List[str]] = None
    get_raw_sample_size: Optional[int] = None
    get_raw_sample_fraction: Optional[float] = 1
    get_raw_streaming: bool = True
    get_raw_max_bytes: int = 100 * 1024 * 1024
    get_raw_shard_sample_count: Optional[int] = None
    get_raw_shard_group_size: Optional[int] = None
    get_raw_max_mbps: Optional[float] = 30.0
    get_raw_preprocessors: List[GetRawPreprocessorConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", self.split.strip())

    def resolve_config_name(self) -> Optional[str]:
        if self.config_template is None:
            return None
        if "{lang}" in self.config_template:
            if not self.lang:
                raise ValueError("preset.lang is required to build config name")
            if len(self.lang) != 1:
                raise ValueError(
                    "preset.lang must contain exactly one value when using {lang} in config_template"
                )
            return self.config_template.format(lang=self.lang[0])
        return self.config_template

    def get_raw_dir(self, output_root: str) -> str:
        return get_raw_dir(output_root, self.name)

    def get_raw_sig_file(self, output_root: str, ensure_dir_exist: bool = False) -> str:
        dir_path = os.path.join(output_root, "signatures")

        if ensure_dir_exist:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        return os.path.join(dir_path, f"{self.name}.parquet")


@dataclass(frozen=True)
class SnapshotPreset:
    name: str
    snapshot_id: str
    split_sizes_chars: Dict[str, int]
    dedup: DedupConfig | None
    excluded_languages: List[str]
    seed: int
    source_configs: Dict[str, SnapshotSourceConfig]


@dataclass(frozen=True)
class TrainingViewPreset:
    name: str
    config: TrainingViewConfig


@dataclass(frozen=True)
class TrainTokenizerPreset:
    name: str
    tokenizer_id: str
    snapshot_id: str
    split: str
    vocab_size: int
    normalization: str = "NFKC"

    def __post_init__(self) -> None:
        normalized = self.normalization.strip().upper()
        if normalized not in {"NFC", "NFKC"}:
            raise ValueError(
                f"unsupported tokenizer normalization: {self.normalization!r}; expected one of ['NFC', 'NFKC']"
            )
        object.__setattr__(self, "normalization", normalized)
