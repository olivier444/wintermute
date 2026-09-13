#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GetRawPreprocessorConfig:
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.kind or not self.kind.strip():
            raise ValueError("preprocessor.kind cannot be empty")
        if not isinstance(self.params, dict):
            raise ValueError("preprocessor.params must be a dict")


@dataclass(frozen=True)
class GetRawConfig:
    dataset_name: str
    config_name: Optional[str]
    split: str
    revision: Optional[str]
    output_dir: str
    language: Optional[List[str]] = None
    language_field: Optional[str] = None
    stat_fields: Optional[List[str]] = None
    exclude_fields: Optional[List[str]] = None
    data_files: Optional[List[str]] = None
    sample_size: Optional[int] = None
    sample_fraction: Optional[float] = None
    seed: int = 1234
    streaming: bool = True
    max_bytes: int = 100 * 1024 * 1024
    output_max_shards: Optional[int] = 200
    hf_shard_sample_count: Optional[int] = None
    hf_shard_group_size: Optional[int] = 5
    max_mbps: Optional[float] = 20.0
    preprocessors: List[GetRawPreprocessorConfig] = field(default_factory=list)

    def validate(self) -> None:
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("sample_size must be > 0")
        if self.sample_fraction is not None:
            if not (0.0 < self.sample_fraction <= 1.0):
                raise ValueError("sample_fraction must be in (0, 1]")
        if self.sample_size is not None and self.sample_fraction is not None:
            raise ValueError("set either sample_size or sample_fraction, not both")
        if self.language is not None:
            if not self.language:
                raise ValueError("language list cannot be empty")
            if any(not lang for lang in self.language):
                raise ValueError("language list cannot contain empty values")
        if self.language_field is not None and not self.language_field:
            raise ValueError("language_field cannot be empty")
        if self.stat_fields is not None:
            if not self.stat_fields:
                raise ValueError("stat_fields cannot be empty")
            if any(not field for field in self.stat_fields):
                raise ValueError("stat_fields cannot contain empty values")
        if self.exclude_fields is not None:
            if not self.exclude_fields:
                raise ValueError("exclude_fields cannot be empty")
            if any(not field for field in self.exclude_fields):
                raise ValueError("exclude_fields cannot contain empty values")
        if self.data_files is not None:
            if not self.data_files:
                raise ValueError("data_files cannot be empty")
            if any(not pattern for pattern in self.data_files):
                raise ValueError("data_files cannot contain empty values")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        if self.output_max_shards is not None and self.output_max_shards <= 0:
            raise ValueError("output_max_shards must be > 0")
        if self.hf_shard_sample_count is not None and self.hf_shard_sample_count <= 0:
            raise ValueError("hf_shard_sample_count must be > 0")
        if self.hf_shard_group_size is not None and self.hf_shard_group_size <= 0:
            raise ValueError("hf_shard_group_size must be > 0")
        if self.max_mbps is not None and self.max_mbps <= 0:
            raise ValueError("max_mbps must be > 0 when set")
        if not isinstance(self.preprocessors, list):
            raise TypeError("preprocessors must be a list")
        for preprocessor in self.preprocessors:
            if not isinstance(preprocessor, GetRawPreprocessorConfig):
                raise TypeError("preprocessors entries must be GetRawPreprocessorConfig instances")
            preprocessor.validate()
