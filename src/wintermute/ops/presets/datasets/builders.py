from __future__ import annotations

from dataclasses import replace
from typing import List, Mapping

from ..model import DatasetPreset

THE_STACK_EXCLUDE_FIELDS = [
    "max_stars_repo_head_hexsha",
    "max_stars_repo_licenses",
    "max_stars_count",
    "max_stars_repo_stars_event_min_datetime",
    "max_stars_repo_stars_event_max_datetime",
    "max_issues_repo_path",
    "max_issues_repo_name",
    "max_issues_repo_head_hexsha",
    "max_issues_repo_licenses",
    "max_issues_count",
    "max_issues_repo_issues_event_min_datetime",
    "max_issues_repo_issues_event_max_datetime",
    "max_forks_repo_path",
    "max_forks_repo_name",
    "max_forks_repo_head_hexsha",
    "max_forks_repo_licenses",
    "max_forks_count",
    "max_forks_repo_forks_event_min_datetime",
    "max_forks_repo_forks_event_max_datetime",
]


def cosmopedia_dataset(*, uid: str, name: str, config_template: str) -> DatasetPreset:
    return DatasetPreset(
        uid=uid,
        name=name,
        dataset_name="HuggingFaceTB/cosmopedia",
        text_fields=("text",),
        get_raw_output_max_shards=600,
        config_template=config_template,
        lang=["en"],
        seed=1719,
        get_raw_stat_fields=["seed_data", "audience", "format"],
        get_raw_exclude_fields=["prompt"],
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    )


def xp3_dataset(
    *,
    uid: str,
    name: str,
    language: str,
    data_files: list[str],
    output_max_shards: int,
) -> DatasetPreset:
    return DatasetPreset(
        uid=uid,
        name=name,
        dataset_name="json",
        text_fields=("inputs", "targets"),
        get_raw_output_max_shards=output_max_shards,
        lang=[language],
        get_raw_data_files=data_files,
    )


def the_stack_dataset(
    *,
    uid: str,
    name: str,
    languages: List[str],
    data_files: List[str],
    output_max_shards: int,
) -> DatasetPreset:
    return DatasetPreset(
        uid=uid,
        name=name,
        dataset_name="bigcode/the-stack-dedup",
        text_fields=("content",),
        get_raw_output_max_shards=output_max_shards,
        lan_field="lang",
        lang=languages,
        get_raw_stat_fields=["lang", "ext"],
        get_raw_exclude_fields=list(THE_STACK_EXCLUDE_FIELDS),
        get_raw_data_files=data_files,
        get_raw_sample_fraction=0.5,
        get_raw_shard_sample_count=1024,
        get_raw_shard_group_size=8,
    )


def expand_dataset_splits(
    preset: DatasetPreset,
    split_aliases: Mapping[str, str],
) -> List[DatasetPreset]:
    if len(split_aliases) == 0:
        raise ValueError("split_aliases cannot be empty")

    normalized_aliases: dict[str, str] = {}
    seen_aliases: set[str] = set()
    for split_name, split_alias in split_aliases.items():
        split_name_value = str(split_name).strip()
        split_alias_value = str(split_alias).strip()
        if not split_name_value:
            raise ValueError("split_aliases cannot contain an empty split name")
        if not split_alias_value:
            raise ValueError("split_aliases cannot contain an empty split alias")
        if split_name_value in normalized_aliases:
            raise ValueError(f"duplicate split name in split_aliases: '{split_name_value}'")
        if split_alias_value in seen_aliases:
            raise ValueError(f"duplicate split alias in split_aliases: '{split_alias_value}'")
        normalized_aliases[split_name_value] = split_alias_value
        seen_aliases.add(split_alias_value)

    return [
        replace(
            preset,
            uid=f"{preset.uid}-{split_alias}",
            name=f"{preset.name}-{split_name}",
            split=split_name,
        )
        for split_name, split_alias in normalized_aliases.items()
    ]
