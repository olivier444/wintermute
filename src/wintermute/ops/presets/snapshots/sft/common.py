from __future__ import annotations

from ..builders import src
from .builders import unitary_sft_split_sizes
from ...model import DedupConfig, SnapshotPreset


def unitary_raw_sft_snapshot(
    name: str,
    *,
    source_id: str,
    total_chars: int,
    seed: int = 1234,
    min_chars: int = 1,
    dedup: DedupConfig | None = None,
) -> SnapshotPreset:
    return SnapshotPreset(
        name=name,
        snapshot_id=name,
        split_sizes_chars=unitary_sft_split_sizes(total_chars),
        dedup=dedup,
        excluded_languages=[],
        seed=seed,
        source_configs={
            source_id: src(1.0, min_chars=min_chars, record_transform=None),
        },
    )
