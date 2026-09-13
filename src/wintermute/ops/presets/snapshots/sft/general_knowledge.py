from __future__ import annotations

from .common import unitary_raw_sft_snapshot


GENERAL_KNOWLEDGE_SFT_SNAPSHOT_PRESETS = [
    # Snapshot: xP3 English Knowledge QA (`xp3-en-knowledge-qa`).
    # Dataset: xP3 English Knowledge QA (`bigscience/xP3`; provider: BigScience).
    # Summary: English closed-book TriviaQA, WebQuestions, and WikiQA answer templates.
    # Purpose: Persist the selected raw xP3 knowledge-QA records; SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `xp3-en-kqa` raw datasource only.
    # Runtime task mapping: TASK_KNOWLEDGE_QA in every knowledge-QA training view.
    # Budget: 49.135M train chars; 2.457M eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "xp3-en-knowledge-qa",
        source_id="xp3-en-kqa",
        total_chars=51_591_642,
    ),
]
