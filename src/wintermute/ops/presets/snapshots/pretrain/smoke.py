from __future__ import annotations

from ..builders import text_src

from ...model import DedupConfig, SnapshotPreset

SMOKE_PRETRAIN_SNAPSHOT_PRESETS = [
    # Snapshot: TinyStories Smoke Test (`tns-smoke`).
    # Purpose: Fast end-to-end pretraining smoke test.
    # Output format: NTP text.
    # Composition: TinyStories only.
    # Budget: 200M train chars; 4M eval chars; 4M test chars.
    # Selection: Deduplicated records of at least 50 characters.
    SnapshotPreset(
        name="tns-smoke",
        snapshot_id="tns-smoke",
        split_sizes_chars={
            "train": int(2e8),  # estimated source size: 2e9 chars
            "eval": int(4e6),
            "test": int(4e6),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=2033,
        source_configs={
            "tns": text_src(1.0, oversampling=1, min_chars=50),  # roneneldan/TinyStories (synthetic stories)
        },
    ),
]
