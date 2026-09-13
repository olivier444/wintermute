from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Any, Dict, Iterator

from wintermute.data.constants import FLD_RECORD_ID
from wintermute.data.record import DataRecord
from wintermute.ml.tokenization.chat_format import WINTERMUTE_CHAT_CONTROL_LITERALS
from wintermute.tools.files import get_materialized_dir, jsonl_dir_iterator
from wintermute.tools.logging import console_log
from wintermute.tools.params import get_bool


_CHAT_CONTROL_TOKEN_PATTERN = "|".join(re.escape(token) for token in WINTERMUTE_CHAT_CONTROL_LITERALS)
_LEADING_SPECIAL_TOKEN_NEWLINES = re.compile(rf"\n+({_CHAT_CONTROL_TOKEN_PATTERN})")
_TRAILING_SPECIAL_TOKEN_NEWLINES = re.compile(rf"({_CHAT_CONTROL_TOKEN_PATTERN})\n+")


def strip_newlines_around_special_tokens(text: str) -> str:
    """Remove legacy structural line feeds adjacent to chat control tokens."""
    text = _LEADING_SPECIAL_TOKEN_NEWLINES.sub(r"\1", text)
    return _TRAILING_SPECIAL_TOKEN_NEWLINES.sub(r"\1", text)


@dataclass(frozen=True)
class MaterializedConfig:
    snapshot_id: str
    split: str
    shard_offset_ratio: float = 0.0
    strip_newlines_around_special_tokens: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "MaterializedConfig":
        payload = dict(data)
        payload.pop("kind", None)
        shard_offset_ratio = float(payload.pop("shard_offset_ratio", 0.0))
        strip_newlines_around_special_tokens = get_bool(
            payload,
            "strip_newlines_around_special_tokens",
            True,
        )
        payload.pop("strip_newlines_around_special_tokens", None)
        if not 0.0 <= shard_offset_ratio <= 1.0:
            raise ValueError(
                f"Invalid materialized shard_offset_ratio: {shard_offset_ratio}. Expected a value between 0 and 1."
            )
        return cls(
            shard_offset_ratio=shard_offset_ratio,
            strip_newlines_around_special_tokens=strip_newlines_around_special_tokens,
            **payload,
        )

    @property
    def name(self) -> str:
        return f"{self.snapshot_id}.{self.split}"

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"kind": "snapshot", "snapshot_id": self.snapshot_id, "split": self.split}
        if self.shard_offset_ratio:
            data["shard_offset_ratio"] = self.shard_offset_ratio
        if not self.strip_newlines_around_special_tokens:
            data["strip_newlines_around_special_tokens"] = False
        return data

class AbstractMaterializedView(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __iter__(self) -> Iterator[DataRecord]: ...


class MaterializedView(AbstractMaterializedView):
    def __init__(
        self,
        output_root: str,
        config: MaterializedConfig,
    ) -> None:
        self.config = config
        self.source_dir = get_materialized_dir(output_root, config.snapshot_id, config.split)

    @property
    def name(self) -> str:
        return self.config.name

    def __iter__(self) -> Iterator[DataRecord]:
        if self.config.shard_offset_ratio:
            console_log(
                "materialized-view",
                f"[{self.name}] starting at shard offset ratio {self.config.shard_offset_ratio:g}",
            )
        for obj in jsonl_dir_iterator(
            self.source_dir,
            shard_offset_ratio=self.config.shard_offset_ratio,
        ):
            record_id = str(obj[FLD_RECORD_ID])
            fields = {
                str(key): value
                for key, value in obj.items()
                if key != FLD_RECORD_ID
            }
            if self.config.strip_newlines_around_special_tokens:
                # Existing large snapshots used newline-delimited chat tokens; migrate string fields at read time
                # so they train with the compact grammar without requiring a costly rematerialization.
                fields = {
                    key: strip_newlines_around_special_tokens(value) if isinstance(value, str) else value
                    for key, value in fields.items()
                }
            yield DataRecord(record_id=record_id, fields=fields)
