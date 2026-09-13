from __future__ import annotations

from ...model import DatasetPreset

REASONING_SFT_DATASET_PRESETS = [
    # Dataset: CoT Collection (`kaist-ai/CoT-Collection`; provider: KAIST AI).
    # Summary: FLAN tasks augmented with explicit rationales and final answers.
    # Natural format: SFT with chain-of-thought.
    # Orientation: general reasoning, multi-task instruction following.
    # Recommended SFT task spec: omitted; the collection mixes extraction, classification,
    # generation, knowledge QA, and problem solving, while the raw `task` field names source datasets.
    # Volume: large (1.84 million English examples across 1,060 tasks).
    DatasetPreset(
        uid="kai-cot",
        name="kaist-cot-collection",
        dataset_name="kaist-ai/CoT-Collection",
        text_fields=("source", "target", "rationale"),
        get_raw_output_max_shards=200,
        config_template="en",
        lang=["en"],
        get_raw_stat_fields=["task", "type"],
        seed=3067,
    ),
    # Dataset: OpenThoughts3-1.2M (`open-thoughts/OpenThoughts3-1.2M`; provider: Open Thoughts).
    # Summary: Multi-domain instruction conversations with long `think` traces, including math and code.
    # Natural format: SFT with chain-of-thought.
    # Orientation: reasoning, mathematics, code.
    # Recommended SFT task spec: `problem_solving` (constant); `domain` is descriptive but not the task.
    # Volume: large (about 1.2 million examples; this preset samples 50%).
    DatasetPreset(
        uid="oth",
        name="open-thoughts-3",
        dataset_name="open-thoughts/OpenThoughts3-1.2M",
        text_fields=("conversations",),
        get_raw_output_max_shards=300,
        lang=["en"],
        get_raw_stat_fields=["domain", "difficulty"],
        get_raw_sample_fraction=0.5,
        seed=3719,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),
]
