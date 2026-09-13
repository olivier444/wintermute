#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random
import re
import time
from pathlib import Path
from dataclasses import asdict
from typing import cast, Any, Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Set, Tuple, runtime_checkable

from wintermute.data.get_raw.config import GetRawConfig
from wintermute.data.get_raw.preprocessors import build_preprocessor
from wintermute.tools.files import JSonLShardWriter, open_read, open_write
from wintermute.tools.huggingface import get_dataset_parquet_urls, is_dataset_script_unsupported_error
from wintermute.tools.logging import console_log
from wintermute.tools.misc import FileController

_STATE_FILENAME = "state.json"
_STATS_FILENAME = "stats.json"


class _ByteRateLimiter:
    def __init__(
        self,
        max_mbps: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._bytes_per_second = max_mbps * 1_000_000 / 8
        self._clock = clock
        self._sleep = sleep
        self._next_allowed_at = clock()

    def consume(self, byte_count: int) -> None:
        if byte_count <= 0:
            return

        now = self._clock()
        next_allowed_at = self._next_allowed_at + byte_count / self._bytes_per_second
        delay = next_allowed_at - now
        if delay > 0:
            self._sleep(delay)
        self._next_allowed_at = max(next_allowed_at, self._clock())


@runtime_checkable
class ShardableItems(Protocol):
    num_shards: int

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ...

    def shard(self, *, num_shards: int, index: int) -> "ShardableItems":
        ...


def _load_state(output_dir: str) -> Dict[str, Any]:
    path = os.path.join(output_dir, _STATE_FILENAME)
    if not os.path.exists(path):
        return {}
    try:
        with open_read(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        console_log("get_raw", f"failed to read state file: {path} ({exc})")
        return {}


def _write_state(
    output_dir: str,
    processed_shard_ids: Set[int],
    writer: "JSonLShardWriter",
    stat_counts: Dict[str, Dict[str, int]],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "processed_shard_ids": sorted(processed_shard_ids),
        "total_records": writer.total_records,
        "total_bytes": writer.total_bytes,
        "shards_written": writer.shard_count,
        "stat_counts": stat_counts,
        "updated_at": time.time(),
    }
    path = os.path.join(output_dir, _STATE_FILENAME)
    tmp_path = f"{path}.tmp"
    with open_write(tmp_path) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _iter_dataset(config: GetRawConfig) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset

    data_files = None
    if config.data_files:
        data_files = {config.split: list(config.data_files)}

    try:
        ds = load_dataset(
            config.dataset_name,
            config.config_name,
            split=config.split,
            revision=config.revision,
            streaming=config.streaming,
            data_files=data_files,
        )
    except RuntimeError as exc:
        if (
            not is_dataset_script_unsupported_error(exc)
            or data_files is not None
            or config.revision is not None
        ):
            raise

        parquet_urls = get_dataset_parquet_urls(
            config.dataset_name,
            config.config_name,
            config.split,
        )
        console_log(
            "get_raw",
            f"falling back to Hugging Face parquet files for {config.dataset_name}"
            f" / {config.config_name or 'default'} / {config.split} ({len(parquet_urls)} shard(s))",
        )
        ds = load_dataset(
            "parquet",
            split=config.split,
            streaming=config.streaming,
            data_files={config.split: parquet_urls},
        )

    return cast(Iterable[Dict[str, Any]], ds)


def _maybe_sample_shards(
    items: Iterable[Dict[str, Any]] | ShardableItems,
    config: GetRawConfig,
    processed_shard_ids: Set[int],
) -> Iterable[Dict[str, Any]]:
    if not config.streaming or config.hf_shard_sample_count is None:
        return items

    if not isinstance(items, ShardableItems):
        console_log("get_raw", "item source is not shardable; skipping shard sampling")
        return items
    
    num_shards = items.num_shards
    if num_shards <= 0:
        console_log("get_raw", "no shard info available; skipping shard sampling")
        return items

    available_indices = [idx for idx in range(num_shards) if idx not in processed_shard_ids]
    if not available_indices:
        console_log("get_raw", "no shards left after excluding processed shards")
        return []

    shard_count = min(config.hf_shard_sample_count, len(available_indices))
    if shard_count < num_shards:
        console_log("get_raw", f"sampling {shard_count} shards out of {num_shards} total shards")
    else:
        console_log("get_raw", f"using all {num_shards} shards")

    if len(available_indices) < num_shards:
        skipped = num_shards - len(available_indices)
        console_log("get_raw", f"skipping {skipped} already-processed shards")

    rng = random.Random(config.seed)
    indices = rng.sample(available_indices, shard_count)
    console_log("get_raw", f"shard indices (sample): {indices[:10]}")

    from datasets import interleave_datasets

    group_size = max(config.hf_shard_group_size or 1, 1)

    def _iter_interleaved_groups() -> Iterator[Dict[str, Any]]:
        for start in range(0, len(indices), group_size):
            group = indices[start : start + group_size]
            console_log("get_raw", f"interleaving shards: {group}")

            shards = [items.shard(num_shards=num_shards, index=idx) for idx in group]
            for item in interleave_datasets(cast(List[Any], shards), seed=config.seed):
                yield item
            processed_shard_ids.update(group)

    return _iter_interleaved_groups()


def _filter_language(
    items: Iterable[Dict[str, Any]],
    language: List[str],
    language_field: str,
) -> Iterator[Dict[str, Any]]:
    allowed = {lang.lower() for lang in language}
    for item in items:
        value = item.get(language_field)
        if isinstance(value, str) and value.lower() in allowed:
            yield item


def _sample_fraction(
    items: Iterable[Dict[str, Any]], fraction: float, seed: int
) -> Iterator[Dict[str, Any]]:
    console_log("get_raw", f"sampling fraction: {fraction}")
    rng = random.Random(seed)
    for item in items:
        if rng.random() < fraction:
            yield item


def _reservoir_sample(
    items: Iterable[Dict[str, Any]], k: int, seed: int
) -> List[Dict[str, Any]]:
    console_log("get_raw", f"reservoir sampling: {k}")
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        if i <= k:
            reservoir.append(item)
        else:
            j = rng.randint(1, i)
            if j <= k:
                reservoir[j - 1] = item
    return reservoir


def _find_next_shard_index(output_dir: str) -> Tuple[int, int, int]:
    if not os.path.isdir(output_dir):
        return 1, 3, 0
    indices: List[int] = []
    widths: List[int] = []
    for name in os.listdir(output_dir):
        match = re.fullmatch(r"data_(\d+)\.jsonl", name)
        if not match:
            match = re.fullmatch(r"data_(\d+)\.jsonl.gz", name)
            
        if match:
            token = match.group(1)
            indices.append(int(token))
            widths.append(len(token))
    if not indices:
        return 1, 3, 0
    max_index = max(indices)
    width = max(max(widths), 3)
    return max_index + 1, width, len(indices)


def _write_jsonl_sharded(
    writer: JSonLShardWriter,
    items: Iterable[Dict[str, Any]],
    stat_fields: Optional[List[str]],
    stat_counts: Dict[str, Dict[str, int]],
    exclude_fields: Optional[List[str]],
    max_mbps: Optional[float],
) -> Tuple[int, int, int]:
    controller = FileController(Path(writer.output_dir))
    rate_limiter = _ByteRateLimiter(max_mbps) if max_mbps is not None else None

    exclude_set = set(exclude_fields or [])
    try:
        for item in items:
            controller.check_pause()

            payload = (
                {key: value for key, value in item.items() if key not in exclude_set}
                if exclude_set
                else item
            )
            bytes_before_write = writer.total_bytes
            if not writer.write(payload):
                break
            if rate_limiter is not None:
                rate_limiter.consume(writer.total_bytes - bytes_before_write)
            if stat_fields:
                for field in stat_fields:
                    value = item.get(field)
                    if value is None:
                        continue
                    field_counts = stat_counts.setdefault(field, {})
                    if isinstance(value, list):
                        for entry in value:
                            if entry is None:
                                continue
                            if isinstance(entry, str):
                                key = entry.lower()
                            else:
                                key = str(entry)
                            field_counts[key] = field_counts.get(key, 0) + 1
                        continue
                    if isinstance(value, str):
                        key = value.lower()
                    else:
                        key = str(value)
                    field_counts[key] = field_counts.get(key, 0) + 1
    finally:
        writer.close()
    return (
        writer.total_records,
        writer.total_bytes,
        writer.shard_count,
    )


def _build_items(
    config: GetRawConfig,
    processed_shard_ids: Set[int],
) -> Tuple[Iterable[Dict[str, Any]], Optional[Dict[str, Any]]]:
    items: Iterable[Dict[str, Any]] = _iter_dataset(config)
    items = _maybe_sample_shards(items, config, processed_shard_ids)
    items = _apply_preprocessors(items, config)
    if config.language and config.language_field:
        items = _filter_language(items, config.language, config.language_field)

    if config.sample_fraction is not None:
        sampled_items = _sample_fraction(items, config.sample_fraction, config.seed)
        sample_note: Optional[Dict[str, Any]] = {
            "mode": "fraction",
            "value": config.sample_fraction,
        }
    elif config.sample_size is not None:
        sampled_items = _reservoir_sample(items, config.sample_size, config.seed)
        sample_note = {"mode": "size", "value": config.sample_size}
    else:
        sampled_items = items
        sample_note = None
    return sampled_items, sample_note


def _apply_preprocessors(
    items: Iterable[Dict[str, Any]],
    config: GetRawConfig,
) -> Iterable[Dict[str, Any]]:
    for preprocessor_config in config.preprocessors:
        pp_params = dict(preprocessor_config.params)
        pp_params.setdefault("seed", config.seed)
        if config.language_field is not None:
            pp_params["language_field"] = config.language_field
        if config.language is not None:
            pp_params["language"] = config.language

        preprocessor = build_preprocessor(
            preprocessor_config.kind,
            pp_params,
        )
        items = preprocessor.preprocess(items)
    return items


def get_raw_dataset(config: GetRawConfig) -> Dict[str, Any]:
    config.validate()
    if config.max_mbps is not None:
        console_log("get_raw", f"pacing stream at {config.max_mbps:g} Mbit/s")
    max_attempts = 3
    backoff_s = 5

    attempt = 1
    while True:
        state = _load_state(config.output_dir)
        processed_shard_ids = set(state.get("processed_shard_ids", []))
        stat_counts = state.get("stat_counts", {})
        if not isinstance(stat_counts, dict):
            stat_counts = {}

        try:
            sampled_items, sample_note = _build_items(config, processed_shard_ids)

            os.makedirs(config.output_dir, exist_ok=True)
            start_index, index_width, existing_shards = _find_next_shard_index(
                config.output_dir
            )
            remaining_shards = None
            if config.output_max_shards is not None:
                remaining_shards = max(config.output_max_shards - existing_shards, 0)

            def _on_shard_complete(writer: JSonLShardWriter) -> None:
                _write_state(config.output_dir, processed_shard_ids, writer, stat_counts)

            writer = JSonLShardWriter(
                config.output_dir,
                config.max_bytes,
                start_index,
                index_width,
                remaining_shards,
                _on_shard_complete,
            )

            (
                count,
                bytes_written,
                shards_written,
            ) = _write_jsonl_sharded(
                writer,
                sampled_items,
                config.stat_fields,
                stat_counts,
                config.exclude_fields,
                config.max_mbps,
            )
            break
        except Exception as exc:
            if attempt >= max_attempts:
                raise

            console_log("get_raw", f"processing error (attempt {attempt}/{max_attempts}): {exc}")
            console_log("get_raw", f"retrying full stream in {backoff_s}s...")
            time.sleep(backoff_s)
            attempt += 1

    config_dict = asdict(config)
    if "language_field" in config_dict:
        config_dict["get_raw_language_field"] = config_dict.pop("language_field")
    manifest = {
        "dataset": config.dataset_name,
        "config_name": config.config_name,
        "split": config.split,
        "revision": config.revision,
        "language": config.language,
        "get_raw_language_field": config.language_field,
        "stat_counts": stat_counts if config.stat_fields else {},
        "streaming": config.streaming,
        "sample": sample_note,
        "records": count,
        "bytes": bytes_written,
        "max_bytes": config.max_bytes,
        "output_max_shards": config.output_max_shards,
        "existing_shards": existing_shards,
        "shards_written": shards_written,
        "start_index": start_index,
        "index_width": index_width,
        "config": config_dict,
    }
    manifest_path = os.path.join(config.output_dir, "manifest.json")
    stats_path = os.path.join(config.output_dir, _STATS_FILENAME)
    os.makedirs(config.output_dir, exist_ok=True)
    with open_write(manifest_path) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open_write(stats_path) as f:
        json.dump({"stat_counts": stat_counts}, f, ensure_ascii=False, indent=2)

    return {
        "output_dir": config.output_dir,
        "index_width": index_width,
        "manifest_path": manifest_path,
        "stats_path": stats_path,
        "records": count,
        "bytes": bytes_written,
        "config": config_dict,
        "stat_counts": stat_counts if config.stat_fields else {},
    }
