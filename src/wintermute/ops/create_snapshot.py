# -*- coding: utf-8 -*-

from wintermute.data.snapshot.config import SnapshotConfig
from wintermute.data.snapshot.snapshot import create_snapshot
from wintermute.data.snapshot.materialize import materialize
from wintermute.ops.materialized_stats import run_materialized_stats
from wintermute.ops.presets import SnapshotPreset


def run_create_snapshot(preset: SnapshotPreset, output_root: str) -> None:
    cfg = SnapshotConfig(
        snapshot_id=preset.snapshot_id,
        output_root=output_root,
        split_sizes_chars=preset.split_sizes_chars,
        dedup=preset.dedup,
        excluded_languages=preset.excluded_languages,
        seed=preset.seed,
        source_configs=preset.source_configs,
    )

    create_snapshot(cfg)
    materialize(cfg)
    run_materialized_stats(
        output_root=output_root,
        snapshot_id=cfg.snapshot_id,
        split=None,
    )
