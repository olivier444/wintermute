# -*- coding: utf-8 -*-

from wintermute.data.get_raw.config import GetRawConfig
from wintermute.data.get_raw.get_raw import get_raw_dataset
from wintermute.data.get_raw.create_uids import create_uids
from wintermute.data.index.signatures import write_signatures
from wintermute.data.index.config import SignatureConfig, IndexConfig
from wintermute.ops.presets import DatasetPreset
from wintermute.data.index.index import create_index
from wintermute.tools.files import get_index_dir
from wintermute.tools.logging import console_log
from typing import List


def run_get_mraw(presets: List[DatasetPreset], output_root: str) -> None:
    for preset in presets:
        console_log("get-raw", f"processing preset {preset.name}")
        run_get_raw(preset, output_root=output_root)


def run_get_raw(preset: DatasetPreset, output_root: str) -> None:
    config_name = preset.resolve_config_name()
    raw_dir = preset.get_raw_dir(output_root)

    cfg = GetRawConfig(
        dataset_name=preset.dataset_name,
        config_name=config_name,
        split=preset.split,
        revision=preset.revision,
        output_dir=raw_dir,
        language=preset.lang,
        language_field=preset.lan_field,
        stat_fields=preset.get_raw_stat_fields,
        exclude_fields=preset.get_raw_exclude_fields,
        data_files=preset.get_raw_data_files,
        sample_size=preset.get_raw_sample_size,
        sample_fraction=preset.get_raw_sample_fraction,
        seed=preset.seed,
        streaming=preset.get_raw_streaming,
        max_bytes=preset.get_raw_max_bytes,
        output_max_shards=preset.get_raw_output_max_shards,
        hf_shard_sample_count=preset.get_raw_shard_sample_count,
        hf_shard_group_size=preset.get_raw_shard_group_size,
        max_mbps=preset.get_raw_max_mbps,
        preprocessors=preset.get_raw_preprocessors,
    )
    get_raw_dataset(cfg)
    create_uids(raw_dir, preset.uid)

    write_signatures(SignatureConfig(
        raw_jsonl_path=preset.get_raw_dir(output_root),
        text_fields=preset.text_fields,
        output_file_path=preset.get_raw_sig_file(output_root, ensure_dir_exist=True)
    ))    

    create_index(IndexConfig(
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
    ))
    console_log("get-raw", f"download and processing complete for preset {preset.name}")
    
