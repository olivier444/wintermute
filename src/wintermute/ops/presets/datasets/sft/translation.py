from __future__ import annotations

from wintermute.data.get_raw.config import GetRawPreprocessorConfig

from ...model import DatasetPreset

TRANSLATION_SFT_DATASET_PRESETS = [
    # Dataset: News Commentary (`Helsinki-NLP/news_commentary`, en-fr; provider: Helsinki-NLP).
    # Summary: Parallel English-French sentences drawn from news commentary.
    # Natural format: SFT (parallel translation).
    # Orientation: English-French translation, news.
    # Recommended SFT task spec: `translation` (constant).
    # Volume: medium (hundreds of thousands of sentence pairs).
    DatasetPreset(
        uid="nwc-ef",
        name="news-commentary-en-fr",
        dataset_name="Helsinki-NLP/news_commentary",
        text_fields=("translation",),
        get_raw_output_max_shards=100,
        config_template="en-fr",
        lang=["en", "fr"],
        get_raw_exclude_fields=["id"],
        seed=2719,
    ),
    # Dataset: AgentLans EN-FR (`agentlans/en-fr`; provider: AgentLans).
    # Summary: English-French pairs annotated for translation quality and readability, filtered locally.
    # Natural format: SFT (parallel translation).
    # Orientation: English-French translation, linguistic quality.
    # Recommended SFT task spec: `translation` (constant).
    # Volume: medium source corpus; only records with quality >= 0.859 are retained.
    DatasetPreset(
        uid="agl-ef-hq",
        name="agentlans-en-fr-hq",
        dataset_name="agentlans/en-fr",
        text_fields=("english", "french"),
        get_raw_output_max_shards=200,
        lang=["en", "fr"],
        get_raw_stat_fields=["source", "translation_quality", "readability_grade"],
        get_raw_exclude_fields=["id"],
        seed=2719,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
        get_raw_preprocessors=[
            GetRawPreprocessorConfig(
                kind="filter",
                params={"field": "translation_quality", "gte": 0.859},
            ),
        ],
    ),
]
