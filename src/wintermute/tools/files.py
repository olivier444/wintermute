# -*- coding: utf-8 -*-

import os
import re
from wintermute.data.constants import *
from typing import Any, Callable, Dict, List, Optional, Iterator, Sequence
import json
from dataclasses import asdict, is_dataclass
import wintermute.data.constants as constants
from pathlib import Path
import gzip
import yaml
from wintermute.data.text_fields import build_display_text
from wintermute.tools.logging import console_log


JSONL_GZ_PATTERN = re.compile(r"data_(\d+)\.jsonl\.gz")


def get_record_text(raw_json_path: str, uid: str, text_fields: tuple[str, ...]):
    """
    Utility method for testing purpose - not optimized for bulk queries
    """
    for obj in jsonl_file_iterator(raw_json_path):
        if obj[constants.FLD_GENERIC_UID] == uid:
            return build_display_text(obj, text_fields)
        
    return None


def iter_jsonl_paths(raw_jsonl_path: str) -> List[str]:
    """
    Return .jsonl.gz shard paths in deterministic order.
    """
    if os.path.isdir(raw_jsonl_path):
        entries: List[str] = []
        for name in os.listdir(raw_jsonl_path):
            if JSONL_GZ_PATTERN.fullmatch(name):
                entries.append(os.path.join(raw_jsonl_path, name))
        entries.sort(key=lambda path: int(file_nr_from_path(path)))
        if not entries:
            raise ValueError(f"no data_*.jsonl.gz files found in {raw_jsonl_path}")
        return entries

    _validate_jsonl_gz_path(raw_jsonl_path)
    return [raw_jsonl_path]


def file_nr_from_path(path: str) -> str:
    name = os.path.basename(path)
    match = JSONL_GZ_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"invalid jsonl.gz shard filename: {path}")
    return match.group(1)


def _validate_jsonl_gz_path(path: str) -> None:
    if not path.lower().endswith(".jsonl.gz"):
        raise ValueError(f"expected a .jsonl.gz file, got: {path}")


def get_index_dir(global_root_path: str) -> str:
    return os.path.join(global_root_path, DIR_INDEX)


def get_raw_dir(global_root_path: str, source_name: str) -> str:
    return os.path.join(global_root_path, DIR_RAW, source_name)


def get_tokenizer_root(global_root_path: str) -> str:
    return os.path.join(global_root_path, DIR_TOKENIZER)


def get_tokenizer_file(global_root_path: str, tokenizer_id: str, ensure_dir_exist: bool = False) -> str:
    dir = os.path.join(get_tokenizer_root(global_root_path), tokenizer_id)
    
    if ensure_dir_exist:
        dp = Path(dir)
        dp.mkdir(parents=True, exist_ok=True)

    return os.path.join(dir, "tokenizer.json")


def get_materialized_dir(global_root_path: str, snapshot_id: str, split: Optional[str] = None) -> str:
    dir = os.path.join(global_root_path, DIR_MATERIALIZED, snapshot_id)
    if split:
        dir = os.path.join(dir, split)
    return dir


def load_jsonl_file(
        input_path: str, 
        id_field: str,
        value_fields: Optional[str | Sequence[str]], 
        container: Dict[str, Any]
    ) -> None:
    
    console_log(
        "load_jsonl_file",
        f"loading {input_path} using fields [{id_field}] / [{_describe_value_fields(value_fields)}]",
    )
    read = 0
    for obj in jsonl_file_iterator(input_path):          
        read += 1
        uid = obj[id_field]
        container[uid] = _extract_jsonl_value(obj, value_fields)
            
    console_log("load_jsonl_file", f"{read:_} records loaded")


def _describe_value_fields(value_fields: Optional[str | Sequence[str]]) -> str:
    if value_fields is None:
        return "*"
    if isinstance(value_fields, str):
        return value_fields
    return ", ".join(value_fields)


def _extract_jsonl_value(
        obj: Dict[str, Any],
        value_fields: Optional[str | Sequence[str]],
    ) -> Any:
    if value_fields is None:
        return dict(obj)

    if isinstance(value_fields, str):
        return obj[value_fields]

    return {field: obj[field] for field in value_fields}


def load_jsonl_line(jsonl_line: str):
    jsonl_line = jsonl_line.strip()
    if not jsonl_line:
        return None
    
    return json.loads(jsonl_line)


def jsonl_dir_iterator(jsonl_dir: str, *, shard_offset_ratio: float = 0.0) -> Iterator[Any]:
    if not 0.0 <= shard_offset_ratio <= 1.0:
        raise ValueError(f"shard_offset_ratio must be between 0 and 1, got {shard_offset_ratio}.")

    files = iter_jsonl_paths(jsonl_dir)
    start_index = int(len(files) * shard_offset_ratio)
    for filepath in files[start_index:]:
        iter = jsonl_file_iterator(filepath)
        for obj in iter:
            yield obj


def jsonl_file_iterator(jsonl_file: str) -> Iterator[Any]:
    _validate_jsonl_gz_path(jsonl_file)
    with open_read(jsonl_file) as f:
        for line in f:
            obj = load_jsonl_line(line)
            if not obj:
                continue

            yield obj


def text_iterator(
        jsonl_directory: str,
        text_field: str) -> Iterator[str]:
    
    for obj in jsonl_dir_iterator(jsonl_directory):
        yield obj[text_field]


def open_read(file_path: str):
    encoding = "utf-8"
    if file_path.lower().endswith(".gz"):
        return gzip.open(file_path, mode="rt", encoding=encoding)
    return open(file_path, "r", encoding=encoding)


def open_write(file_path: str):
    encoding = "utf-8"
    if file_path.lower().endswith(".gz"):
        return gzip.open(file_path, mode="wt", encoding=encoding)
    return open(file_path, "w", encoding=encoding)
    

def json_save(obj, directory: str, file_name: str) -> None:
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        obj = obj.to_dict()
    elif is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)

    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    with (dir_path / file_name).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)    

 
def json_load(directory: str, file_name: str) -> Dict[str, Any]:
    path = Path(directory) / file_name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def yaml_save(obj, directory: str, file_name: str) -> None:
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        obj = obj.to_dict()
    elif is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)

    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    with (dir_path / file_name).open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            obj,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def yaml_load(directory: str, file_name: str) -> Any:
    path = Path(directory) / file_name
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class JSonLShardWriter:
    def __init__(
        self,
        output_dir: str,
        max_bytes: int,
        start_index: int,
        width: int,
        output_max_shards: Optional[int] = None,
        on_shard_complete: Optional[Callable[["JSonLShardWriter"], None]] = None,
    ) -> None:
        self.output_dir = output_dir
        self.max_bytes = max_bytes
        self.output_max_shards = output_max_shards
        self._index = start_index
        self._width = max(width, 4)
        self._records_in_shard = 0
        self._bytes_in_shard = 0
        self._data_fh = None
        self._on_shard_complete = on_shard_complete
        self.total_records = 0
        self.total_bytes = 0
        self.shard_count = 0
        os.makedirs(output_dir, exist_ok=True)

    def _close_current_shard(self) -> None:
        if self._data_fh:
            self._data_fh.close()
            self._data_fh = None
            if self._on_shard_complete:
                self._on_shard_complete(self)

    def _open_next(self) -> bool:
        if (
            self.output_max_shards is not None
            and self.shard_count >= self.output_max_shards
        ):
            return False

        self._close_current_shard()

        data_name = f"data_{self._index:0{self._width}d}.jsonl.gz"
        self.data_path = os.path.join(self.output_dir, data_name)

        console_log("shard-writer", f"writing to [{self.data_path}]")
        self._data_fh = open_write(self.data_path)
        self._records_in_shard = 0
        self._bytes_in_shard = 0
        self._index += 1
        self.shard_count += 1
        return True

    def write(self, item: Dict[str, Any]) -> bool:
        if self._data_fh is None:
            if not self._open_next():
                return False
        line = json.dumps(item, ensure_ascii=False)
        line_bytes = len(line.encode("utf-8")) + 1
        if self._bytes_in_shard > 0 and self._bytes_in_shard + line_bytes > self.max_bytes:
            if not self._open_next():
                return False
            
        
        assert self._data_fh is not None
        self._data_fh.write(line + "\n")
        self._records_in_shard += 1
        self._bytes_in_shard += line_bytes
        self.total_records += 1
        self.total_bytes += line_bytes
        return True

    def close(self) -> None:
        self._close_current_shard()
