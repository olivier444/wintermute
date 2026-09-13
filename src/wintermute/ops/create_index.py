# -*- coding: utf-8 -*-

from typing import Iterable

from wintermute.ops.presets import DatasetPreset
from wintermute.data.index.config import IndexConfig
from wintermute.data.index.index import create_index
from wintermute.tools.files import get_index_dir


def run_create_index(preset: DatasetPreset, output_root: str) -> None:
    raw_dir = preset.get_raw_dir(output_root)
    config_name = preset.resolve_config_name()

    cfg = IndexConfig(
        source_id=preset.uid,
        source_label=preset.name,
        dataset_name=preset.dataset_name,
        config_name=config_name,
        split=preset.split,
        revision=preset.revision,
        raw_jsonl_path=raw_dir,
        sig_path=preset.get_raw_sig_file(output_root),
        text_fields=preset.text_fields,
        lan_field=preset.lan_field,
        output_root_path=get_index_dir(output_root),
    )
    create_index(cfg)


def run_create_index_group(presets: Iterable[DatasetPreset], output_root: str) -> None:
    for preset in presets:
        run_create_index(preset, output_root)
