#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from wintermute.tools.files import get_index_dir, get_materialized_dir, get_tokenizer_file

SUPPORTED_TOKENIZER_NORMALIZATIONS = {"NFC", "NFKC"}


@dataclass(frozen=True)
class TokenizeConfig:
    tokenizer_id: str
    output_root: str
    training_snapshot_id: str
    training_snapshot_split: str
    vocab_size: int
    normalization: str = "NFKC"

    def __post_init__(self) -> None:
        normalized = self.normalization.strip().upper()
        if normalized not in SUPPORTED_TOKENIZER_NORMALIZATIONS:
            raise ValueError(
                f"unsupported tokenizer normalization: {self.normalization!r}; "
                f"expected one of {sorted(SUPPORTED_TOKENIZER_NORMALIZATIONS)}"
            )
        object.__setattr__(self, "normalization", normalized)

    def get_index_path(self) -> str:
        return get_index_dir(self.output_root)

    def get_training_data_path(self) -> str:
        return get_materialized_dir(self.output_root, self.training_snapshot_id, self.training_snapshot_split)

    def get_tokenizer_file(self, ensure_dir_exist: bool = False) -> str:
        return get_tokenizer_file(self.output_root, self.tokenizer_id, ensure_dir_exist)
