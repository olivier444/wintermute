from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen


_SCRIPT_UNSUPPORTED_MARKER = "Dataset scripts are no longer supported"


def is_dataset_script_unsupported_error(exc: Exception) -> bool:
    return _SCRIPT_UNSUPPORTED_MARKER in str(exc)


def _load_dataset_server_parquet_payload(dataset_name: str) -> Dict[str, Any]:
    url = f"https://datasets-server.huggingface.co/parquet?dataset={quote(dataset_name, safe='')}"
    with urlopen(url, timeout=30) as response:
        payload = json.load(response)

    if not isinstance(payload, dict):
        raise ValueError(
            f"unexpected dataset-server parquet payload type for {dataset_name}: "
            f"{type(payload).__name__}"
        )

    return payload


def _build_convert_parquet_url(
    dataset_name: str,
    config_name: str,
    split: str,
    shard_index: int,
) -> str:
    return (
        f"https://huggingface.co/datasets/{dataset_name}"
        f"/resolve/refs%2Fconvert%2Fparquet/{quote(config_name, safe='')}/{quote(split, safe='')}"
        f"/{shard_index:04d}.parquet"
    )


def _url_exists(url: str) -> bool:
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=30):
        return True


def _probe_convert_parquet_urls(
    dataset_name: str,
    config_name: str,
    split: str,
    *,
    max_shards: int = 2048,
) -> List[str]:
    candidate_splits = [split]
    if not split.startswith("partial-"):
        candidate_splits.append(f"partial-{split}")

    for candidate_split in candidate_splits:
        urls: List[str] = []
        for shard_index in range(max_shards):
            url = _build_convert_parquet_url(dataset_name, config_name, candidate_split, shard_index)
            try:
                if not _url_exists(url):
                    break
            except Exception:
                break
            urls.append(url)

        if urls:
            return urls

    return []


def get_dataset_parquet_index(dataset_name: str) -> Dict[str, Dict[str, List[str]]]:
    payload = _load_dataset_server_parquet_payload(dataset_name)
    parquet_files = payload.get("parquet_files")
    if not isinstance(parquet_files, list):
        raise ValueError(f"missing parquet_files in dataset-server payload for {dataset_name}")

    normalized: Dict[str, Dict[str, List[str]]] = {}
    for entry in parquet_files:
        if not isinstance(entry, Mapping):
            continue

        config_name = entry.get("config")
        split_name = entry.get("split")
        url = entry.get("url")
        if not isinstance(config_name, str) or not isinstance(split_name, str) or not isinstance(url, str):
            continue

        normalized.setdefault(config_name, {}).setdefault(split_name, []).append(url)

    if not normalized:
        raise ValueError(f"no parquet files listed for dataset {dataset_name}")

    return normalized


def get_dataset_parquet_urls(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
) -> List[str]:
    resolved_config_name = config_name
    parquet_index: Dict[str, Dict[str, List[str]]] | None = None
    try:
        parquet_index = get_dataset_parquet_index(dataset_name)
    except Exception:
        parquet_index = None

    if parquet_index is not None:
        if resolved_config_name is None:
            if "default" in parquet_index:
                resolved_config_name = "default"
            elif len(parquet_index) == 1:
                resolved_config_name = next(iter(parquet_index))
            else:
                available_configs = ", ".join(sorted(parquet_index))
                raise ValueError(
                    f"dataset {dataset_name} exposes multiple parquet configs; choose one of: {available_configs}"
                )

        split_map = parquet_index.get(resolved_config_name)
        if split_map is None:
            available_configs = ", ".join(sorted(parquet_index))
            raise ValueError(
                f"unknown parquet config '{resolved_config_name}' for dataset {dataset_name}; "
                f"available configs: {available_configs}"
            )

        urls = split_map.get(split)
        if urls is None:
            available_splits = ", ".join(sorted(split_map))
            raise ValueError(
                f"unknown parquet split '{split}' for dataset {dataset_name}/{resolved_config_name}; "
                f"available splits: {available_splits}"
            )

        return list(urls)

    if resolved_config_name is None:
        raise ValueError(
            f"unable to discover parquet files for dataset {dataset_name}: "
            "the datasets-server parquet endpoint failed and no explicit config_name was provided"
        )

    probed_urls = _probe_convert_parquet_urls(dataset_name, resolved_config_name, split)
    if probed_urls:
        return probed_urls

    raise ValueError(
        f"unable to discover parquet files for dataset {dataset_name}/{resolved_config_name}/{split}"
    )


def get_parquet_split_names(
    dataset_name: str,
    config_name: Optional[str],
) -> List[str]:
    parquet_index = get_dataset_parquet_index(dataset_name)

    resolved_config_name = config_name
    if resolved_config_name is None:
        if "default" in parquet_index:
            resolved_config_name = "default"
        elif len(parquet_index) == 1:
            resolved_config_name = next(iter(parquet_index))
        else:
            return []

    split_map = parquet_index.get(resolved_config_name)
    if split_map is None:
        return []

    return sorted(split_map)


def get_parquet_config_names(dataset_name: str) -> List[str]:
    return sorted(get_dataset_parquet_index(dataset_name))
