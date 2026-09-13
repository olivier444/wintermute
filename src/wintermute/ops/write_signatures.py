# -*- coding: utf-8 -*-

from wintermute.data.index.config import SignatureConfig
from wintermute.data.index.signatures import write_signatures
from wintermute.ops.presets import DatasetPreset


def run_write_signatures(preset: DatasetPreset, output_root: str) -> None:
    write_signatures(SignatureConfig(
        raw_jsonl_path=preset.get_raw_dir(output_root),
        text_fields=preset.text_fields,
        output_file_path=preset.get_raw_sig_file(output_root, ensure_dir_exist=True)
    ))
