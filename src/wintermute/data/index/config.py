#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class SignatureConfig:
    raw_jsonl_path: str
    output_file_path: str
    text_fields: tuple[str, ...]
    ngram_size: int = 5
    num_hash: int = 128

@dataclass(frozen=True)
class IndexConfig:
    source_id: str
    source_label: str
    dataset_name: str
    config_name: Optional[str]
    split: str
    revision: Optional[str]
    raw_jsonl_path: str
    sig_path: str
    text_fields: tuple[str, ...]
    lan_field: Optional[str]
    output_root_path: str
    num_hash: int = 128
    band_size: int = 8
