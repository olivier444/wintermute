from __future__ import annotations

from ...model import DatasetPreset

SPECIALIZED_PRETRAIN_DATASET_PRESETS = [
    # Dataset: PG-19 (`emozilla/pg19`; provider: Project Gutenberg, republished by emozilla).
    # Summary: English books published before 1919, selected from Project Gutenberg.
    # Natural format: classic NTP (long-form books).
    # Orientation: foundation, long-context literature.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: medium (13,700 training books; about 11.4B characters).
    DatasetPreset(
        uid="pg",
        name="pg19",
        dataset_name="emozilla/pg19",
        text_fields=("text",),
        get_raw_output_max_shards=200,
        config_template="default",
    ),
    # Dataset: OpenWebMath (`open-web-math/open-web-math`; provider: OpenWebMath).
    # Summary: High-quality English mathematical web text with LaTeX extracted from Common Crawl.
    # Natural format: classic NTP (web documents).
    # Orientation: foundation, mathematics, STEM.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: very large (6.3M documents and 14.7B tokens; this preset samples 50%).
    DatasetPreset(
        uid="owm",
        name="open-web-math",
        dataset_name="open-web-math/open-web-math",
        text_fields=("text",),
        get_raw_output_max_shards=300,
        lang=["en"],
        get_raw_sample_fraction=0.5,
        seed=1919,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),
    # Dataset: Nemotron CC Math 4plus (`nvidia/Nemotron-CC-Math-v1`, 4plus; provider: NVIDIA).
    # Summary: Highest-quality tier of math-heavy Common Crawl text with equations/code preserved and normalized.
    # Natural format: classic NTP (math documents, tutorials, exercises and worked solutions).
    # Orientation: specialized pretraining, mathematics and STEM.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: about 52.3B tokens in 46 HF shards. Sampling 5 shards and retaining 33% of records targets
    # roughly 1.9-2.0B tokens (actual volume depends on shard token density).
    DatasetPreset(
        uid="nem-math4p",
        name="nemotron-cc-math-4plus",
        dataset_name="nvidia/Nemotron-CC-Math-v1",
        config_template="4plus",
        text_fields=("text",),
        get_raw_output_max_shards=300,
        get_raw_sample_fraction=0.33,
        seed=6239,
        get_raw_shard_sample_count=5,
        get_raw_shard_group_size=8,
    ),    
    # Dataset: Ultra-FineWeb-L3 English QA (`openbmb/Ultra-FineWeb-L3`; provider: OpenBMB).
    # Summary: High-quality web documents transformed into original-text + multiple synthetic Q&A pairs.
    # Natural format: NTP over the synthesized `content` field.
    # Orientation: capability-oriented late-stage pretraining, contextual retrieval and comprehension.
    # Recommended SFT task spec: not applicable; intended here as pretraining/CPT data.
    # Volume: about 245B MiniCPM5 tokens across roughly 320 HF shards. Sampling
    # 70 physical shards and retaining 17% of records may yield about 9.0B tokens.
    DatasetPreset(
        uid="ufw-l3-qa",
        name="ultra-fineweb-l3-en-qa",
        dataset_name="openbmb/Ultra-FineWeb-L3",
        config_template="Ultra-FineWeb-L3-en-QA-Synthetic",
        text_fields=("content",),
        get_raw_output_max_shards=400,
        get_raw_exclude_fields=["uid", "style"],
        get_raw_sample_fraction=0.17,
        seed=6263,
        get_raw_shard_sample_count=70,
        get_raw_shard_group_size=8,
    ),
]
