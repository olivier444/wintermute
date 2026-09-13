from __future__ import annotations

from typing import Callable, Dict, List, Sequence, TypeVar

from .datasets import DATASET_PRESETS
from .model import (
    DatasetPreset,
    SnapshotPreset,
    TrainingViewPreset,
    TrainTokenizerPreset,
)
from .snapshots import SNAPSHOT_PRESETS
from .tokenizers import TRAIN_TOKENIZER_PRESETS
from .views import TRAINING_VIEW_PRESETS

T = TypeVar("T")


def _list_presets(presets: Sequence[T], *, key_fn: Callable[[T], str]) -> Dict[str, T]:
    return {key_fn(preset): preset for preset in presets}


def _get_preset(
    presets: Sequence[T],
    key: str,
    *,
    kind: str,
    alias_fns: Sequence[Callable[[T], str]],
) -> T:
    normalized = key.strip().lower()
    for preset in presets:
        if any(alias_fn(preset).lower() == normalized for alias_fn in alias_fns):
            return preset
    raise ValueError(f"unknown {kind}: {normalized}")


def list_dataset_presets() -> Dict[str, DatasetPreset]:
    return _list_presets(DATASET_PRESETS, key_fn=lambda preset: preset.name)


def get_dataset_presets(keys: str) -> List[DatasetPreset]:
    return [get_dataset_preset(key.strip()) for key in keys.split(",")]


def get_dataset_preset(key: str) -> DatasetPreset:
    return _get_preset(
        DATASET_PRESETS,
        key,
        kind="preset",
        alias_fns=(lambda preset: preset.name, lambda preset: preset.uid),
    )


def list_snapshot_presets() -> Dict[str, SnapshotPreset]:
    return _list_presets(SNAPSHOT_PRESETS, key_fn=lambda preset: preset.name)


def get_snapshot_presets(keys: str) -> List[SnapshotPreset]:
    return [get_snapshot_preset(key.strip()) for key in keys.split(",")]


def get_snapshot_preset(key: str) -> SnapshotPreset:
    return _get_preset(
        SNAPSHOT_PRESETS,
        key,
        kind="snapshot preset",
        alias_fns=(lambda preset: preset.name, lambda preset: preset.snapshot_id),
    )


def list_training_view_presets() -> Dict[str, TrainingViewPreset]:
    return _list_presets(TRAINING_VIEW_PRESETS, key_fn=lambda preset: preset.name)


def get_training_view_preset(key: str) -> TrainingViewPreset:
    return _get_preset(
        TRAINING_VIEW_PRESETS,
        key,
        kind="training view preset",
        alias_fns=(lambda preset: preset.name,),
    )


def list_train_tokenizer_presets() -> Dict[str, TrainTokenizerPreset]:
    return _list_presets(TRAIN_TOKENIZER_PRESETS, key_fn=lambda preset: preset.name)


def get_train_tokenizer_preset(key: str) -> TrainTokenizerPreset:
    return _get_preset(
        TRAIN_TOKENIZER_PRESETS,
        key,
        kind="train tokenizer preset",
        alias_fns=(lambda preset: preset.name, lambda preset: preset.tokenizer_id),
    )


def list_presets() -> Dict[str, DatasetPreset]:
    return list_dataset_presets()


def get_preset(key: str) -> DatasetPreset:
    return get_dataset_preset(key)
