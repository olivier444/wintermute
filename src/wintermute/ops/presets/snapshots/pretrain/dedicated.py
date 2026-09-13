from __future__ import annotations

from ..builders import text_src

from ...model import DedupConfig, SnapshotPreset


DEDICATED_PRETRAIN_SNAPSHOT_PRESETS = [
    # Snapshot: DCLM Baseline Text (`dclm-baseline-20b`).
    # Purpose: Standalone English web pretraining from the locally sampled DCLM Baseline source.
    # Output format: NTP text.
    # Composition: DCLM Baseline English web only.
    # Budget: 64B train chars, leaving source capacity for the separate 10M-char DCLM evaluation snapshot.
    # Selection: Deduplicated records of at least 200 characters, without oversampling.
    SnapshotPreset(
        name="dclm-baseline-20b",
        snapshot_id="dclm-baseline-20b",
        split_sizes_chars={
            "train": int(16e9 * 4),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=2741,
        source_configs={
            "dclm": text_src(1.0, oversampling=1, min_chars=200),
        },
    ),
    # Snapshot: Nemotron CC Math 4plus (`nem-math4p`).
    # Purpose: Standalone mathematics and STEM continued-pretraining data.
    # Output format: NTP text.
    # Composition: NVIDIA Nemotron CC Math 4plus only.
    # Budget: Up to 8B train chars, covering the complete locally sampled source.
    # Selection: Deduplicated records of at least 200 characters, without oversampling.
    SnapshotPreset(
        name="nem-math4p",
        snapshot_id="nem-math4p",
        split_sizes_chars={
            "train": 6_322_370_847,
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=6239,
        source_configs={
            "nem-math4p": text_src(1.0, oversampling=1, min_chars=200),
        },
    ),
    # Snapshot: Ultra-FineWeb L3 English QA (`ufw-l3-qa`).
    # Purpose: Standalone contextual-retrieval and comprehension continued-pretraining data.
    # Output format: NTP text.
    # Composition: OpenBMB Ultra-FineWeb-L3 English synthetic QA content only.
    # Budget: Up to 36B train chars, covering the complete locally sampled source.
    # Selection: Deduplicated records of at least 200 characters, without oversampling.
    SnapshotPreset(
        name="ufw-l3-qa",
        snapshot_id="ufw-l3-qa",
        split_sizes_chars={
            "train": 24_830_000_000,
            "eval": 26_904_598,
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=6263,
        source_configs={
            "ufw-l3-qa": text_src(1.0, field="content", oversampling=1, min_chars=200),
        },
    ),
    # Snapshot: FineWeb2-HQ French Text (`fw2-fr-txt`).
    # Purpose: Standalone French web pretraining from the locally sampled FineWeb2-HQ source.
    # Output format: NTP text.
    # Composition: FineWeb2-HQ French only.
    # Budget: 36B train chars; 10M eval chars; 50M test chars.
    # Selection: Deduplicated records of at least 200 characters, without oversampling.
    SnapshotPreset(
        name="fw2-fr-txt",
        snapshot_id="fw2-fr-txt",
        split_sizes_chars={
            "train": int(9e9 * 4),
            "eval": int(1e7),
            "test": int(5e7),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=1931,
        source_configs={
            "fw2-fr": text_src(1.0, oversampling=1, min_chars=200),
        },
    ),
]
