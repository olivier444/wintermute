from __future__ import annotations

from ...model import DedupConfig
from .common import unitary_raw_sft_snapshot

REASONING_SFT_SNAPSHOT_PRESETS = [
    # Snapshot: CoT Collection (`kaist-cot-collection`).
    # Dataset: CoT Collection (`kaist-ai/CoT-Collection`; provider: KAIST AI).
    # Summary: FLAN tasks augmented with explicit rationales and final answers.
    # Purpose: Persist raw CoT Collection triplets for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: CoT Collection source/target/rationale triplets only.
    # Runtime task mapping: Omitted because the collection mixes extraction, classification,
    # generation, knowledge QA, and problem solving without a reviewed normalization map.
    # Budget: 2.003B train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Deduplicated records with a minimum length of 20 characters.
    unitary_raw_sft_snapshot(
        "kaist-cot-collection",
        source_id="kai-cot",
        total_chars=2_012_802_726,
        seed=3067,
        min_chars=20,
        dedup=DedupConfig(),
    ),
    # Snapshot: OpenThoughts3 (`open-thoughts-3`).
    # Dataset: OpenThoughts3-1.2M (`open-thoughts/OpenThoughts3-1.2M`; provider: Open Thoughts).
    # Summary: Multi-domain instruction conversations with long `think` traces, including math and code.
    # Purpose: Persist raw OpenThoughts conversations for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `oth` raw datasource only.
    # Runtime task mapping: TASK_PROBLEM_SOLVING for every emitted assistant turn.
    # Budget: 31.004B train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once; task-facing views extract closed think tags.
    unitary_raw_sft_snapshot(
        "open-thoughts-3",
        source_id="oth",
        total_chars=31_014_408_293,
        seed=3719,
    ),
    # Snapshot: xP3 English Problem Solving (`xp3-en-problem-solving`).
    # Dataset: xP3 English Problem Solving (`bigscience/xP3`; provider: BigScience).
    # Summary: English ARC-Challenge and WIQA scientific or procedural inference templates.
    # Purpose: Persist raw xP3 problem-solving records for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `xp3-en-ps` raw datasource only.
    # Runtime task mapping: TASK_PROBLEM_SOLVING in every task-facing view.
    # Budget: 107.519M train chars; 5.376M eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "xp3-en-problem-solving",
        source_id="xp3-en-ps",
        total_chars=112_894_937,
    ),
]
