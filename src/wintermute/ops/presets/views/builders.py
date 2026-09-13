from __future__ import annotations

from wintermute.data.iterate.dataclasses import (
    CompositeTrainingSource,
    CompositeTrainingViewConfig,
    SnapshotTrainingViewConfig,
    TrainingViewConfig,
)
from wintermute.data.snapshot.materialized_iterable import MaterializedConfig
from wintermute.data.transform import RecordTransformConfig

from ..model import TrainingViewPreset


def _snapshot_view_config(
    snapshot_id: str,
    *,
    split: str = "train",
    include_prompt_in_loss: bool = False,
    shard_offset_ratio: float = 0.0,
    record_transform: RecordTransformConfig | None = None,
) -> SnapshotTrainingViewConfig:
    return SnapshotTrainingViewConfig(
        materialized_config=MaterializedConfig(
            snapshot_id=snapshot_id,
            split=split,
            shard_offset_ratio=shard_offset_ratio,
        ),
        include_prompt_in_loss=include_prompt_in_loss,
        record_transform=record_transform,
    )


def _view_component(
    snapshot_id: str,
    weight: float,
    *,
    split: str = "train",
    max_restarts: int = 0,
    include_prompt_in_loss: bool = False,
    shard_offset_ratio: float = 0.0,
    record_transform: RecordTransformConfig | None = None,
) -> CompositeTrainingSource:
    return CompositeTrainingSource(
        weight=weight,
        max_restarts=max_restarts,
        view_config=_snapshot_view_config(
            snapshot_id,
            split=split,
            include_prompt_in_loss=include_prompt_in_loss,
            shard_offset_ratio=shard_offset_ratio,
            record_transform=record_transform,
        ),
    )


def _composite_view_config(
    sources: list[CompositeTrainingSource],
    *,
    seed: int,
) -> CompositeTrainingViewConfig:
    return CompositeTrainingViewConfig(
        seed=seed,
        mix_unit="loss_tokens",
        sources=sources,
    )


def _view_preset(name: str, config: TrainingViewConfig) -> TrainingViewPreset:
    return TrainingViewPreset(name=name, config=config)
