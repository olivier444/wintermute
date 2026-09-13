from __future__ import annotations

from ...model import DedupConfig
from .common import unitary_raw_sft_snapshot

TRANSLATION_SFT_SNAPSHOT_PRESETS = [
    # Snapshot: AgentLans EN-FR HQ (`agentlans-en-fr-hq`).
    # Dataset: AgentLans EN-FR (`agentlans/en-fr`; provider: AgentLans).
    # Summary: English-French pairs annotated for translation quality and readability, filtered locally.
    # Purpose: Persist raw English-French pairs for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: High-quality AgentLans English-French parallel data.
    # Runtime task mapping: TASK_TRANSLATION for both directions in every task-facing view.
    # Budget: 313.176M train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Deduplicated source records; deterministic prompt wording varies by record id.
    unitary_raw_sft_snapshot(
        "agentlans-en-fr-hq",
        source_id="agl-ef-hq",
        total_chars=323_175_574,
        seed=2719,
        dedup=DedupConfig(),
    ),
    # Snapshot: News Commentary EN-FR (`news-commentary-en-fr`).
    # Dataset: News Commentary (`Helsinki-NLP/news_commentary`, en-fr; provider: Helsinki-NLP).
    # Summary: Parallel English-French sentences drawn from news commentary.
    # Purpose: Persist raw news-domain parallel text for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `nwc-ef` raw datasource only.
    # Runtime task mapping: TASK_TRANSLATION for both directions in every task-facing view.
    # Budget: 65.087M train chars; 3.254M eval chars; together they cover all raw chars.
    # Selection: Every indexed source record exactly once; deterministic prompt wording varies by record id.
    unitary_raw_sft_snapshot(
        "news-commentary-en-fr",
        source_id="nwc-ef",
        total_chars=68_341_153,
        seed=2722,
    ),
]
