from __future__ import annotations

from ...model import DatasetPreset

FOUNDATION_PRETRAIN_DATASET_PRESETS = [
    # Dataset: Wikimedia Wikipedia (`wikimedia/wikipedia`, 20231101.en; provider: Wikimedia Foundation).
    # Summary: Cleaned full articles from the English Wikipedia dump.
    # Natural format: classic NTP (continuous article text).
    # Orientation: foundation, general knowledge.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: large (about 6.4 million English articles).
    DatasetPreset(
        uid="wk",
        name="wikipedia",
        dataset_name="wikimedia/wikipedia",
        text_fields=("text",),
        get_raw_output_max_shards=400,
        config_template="20231101.{lang}",
        lang=["en"],
    ),
    # Dataset: FineWeb-Edu (`HuggingFaceFW/fineweb-edu`; provider: Hugging Face FineData).
    # Summary: Educational English web pages selected from FineWeb by a quality classifier.
    # Natural format: classic NTP (web documents).
    # Orientation: foundation, educational web.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: very large (1.3T tokens; this preset samples 10%).
    DatasetPreset(
        uid="fwe-2",
        name="fineweb-edu-2",
        dataset_name="HuggingFaceFW/fineweb-edu",
        text_fields=("text",),
        get_raw_output_max_shards=600,
        config_template="default",
        lan_field="language",
        lang=["en"],
        get_raw_stat_fields=["language"],
        get_raw_sample_fraction=0.1,
        seed=1219,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),
    # Dataset: DCLM-Baseline (`mlfoundations/dclm-baseline-1.0-parquet`; provider: DCLM Team).
    # Summary: Curated English Common Crawl documents filtered and deduplicated for language-model pretraining.
    # Natural format: classic NTP (web documents).
    # Orientation: foundation, high-quality general web.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: very large (about 4T tokens in 27,938 shards; this preset reads 420 shards and keeps 1/3 of records,
    # for roughly the same retained volume as 140 full shards while covering about 3x more physical shards).
    DatasetPreset(
        uid="dclm",
        name="dclm-baseline-20b",
        dataset_name="mlfoundations/dclm-baseline-1.0-parquet",
        text_fields=("text",),
        get_raw_output_max_shards=1000,
        revision="817d6752765f6a41261085171dd546b104f60626",
        lan_field="language",
        lang=["en"],
        get_raw_stat_fields=["language"],
        get_raw_sample_fraction=0.33,
        seed=2741,
        get_raw_shard_sample_count=420,
        get_raw_shard_group_size=8,
    ),
    # Dataset: Wikimedia Wikipedia (`wikimedia/wikipedia`, 20231101.fr; provider: Wikimedia Foundation).
    # Summary: Cleaned full articles from the French Wikipedia dump.
    # Natural format: classic NTP (continuous article text).
    # Orientation: foundation, general knowledge, French.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: large (about 2.6 million French articles).
    DatasetPreset(
        uid="wk-fr",
        name="wikipedia-fr",
        dataset_name="wikimedia/wikipedia",
        text_fields=("text",),
        get_raw_output_max_shards=400,
        config_template="20231101.{lang}",
        lang=["fr"],
        seed=7744,
    ),
    # Dataset: FineWeb2-HQ (`epfml/FineWeb2-HQ`, fra_Latn; provider: EPFL MLO).
    # Summary: High-quality French web documents selected from FineWeb2 by a model-based classifier.
    # Natural format: classic NTP (web documents).
    # Orientation: foundation, high-quality web, French.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: very large (part of a 6 TB, 20-language corpus; this preset samples 50%).
    DatasetPreset(
        uid="fw2-fr",
        name="FineWeb2-HQ-fr",
        dataset_name="epfml/FineWeb2-HQ",
        text_fields=("text",),
        get_raw_output_max_shards=400,
        config_template="fra_Latn",
        lan_field="language",
        lang=["fra"],
        get_raw_stat_fields=["lang"],
        get_raw_exclude_fields=["id", "date", "dump", "embeddings", "file_path"],
        get_raw_sample_fraction=0.5,
        seed=1931,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),
]
