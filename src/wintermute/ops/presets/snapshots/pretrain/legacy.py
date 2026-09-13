from __future__ import annotations

from ..builders import chat_to_sft_src, text_src

from ...model import DedupConfig, SnapshotPreset

LEGACY_PRETRAIN_SNAPSHOT_PRESETS = [
    # Snapshot: Default Legacy Text (`default4-txt`).
    # Purpose: Historical broad pretraining mixture retained under the structured chat representation.
    # Output format: NTP text plus structured prompt/completion records.
    # Composition: English web, books, synthetic education, and assistant chat.
    # Budget: 24B train chars; staged eval/test splits from 0.5M to 50M chars.
    # Selection: Deduplicated weighted mixture with source-specific length limits.
    SnapshotPreset(
        name="default-old-txt",
        snapshot_id="default4-txt",
        split_sizes_chars={
            "train": int(50e6 * 4 * 20 * 6),
            "eval1": int(500e3),
            "eval2": int(2e6),
            "eval3": int(1e7),
            "test1": int(10e6),
            "test2": int(20e6),
            "test3": int(50e6),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=17,
        source_configs={
            "wk": text_src(0.15),  # wikimedia/wikipedia (English encyclopedia)
            "fwe-2": text_src(0.3),  # HuggingFaceFW/fineweb-edu (educational web)
            "fwe": text_src(0.2),  # HuggingFaceFW/fineweb-edu (educational web)
            "pg": text_src(0.02, min_chars=800),  # emozilla/pg19 (public-domain books)
            "ckh": text_src(0.01, oversampling=3, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic educational text)
            "cox": text_src(0.03, oversampling=2, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic textbooks)
            "cwh": text_src(0.06, oversampling=2, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic procedural text)
            "cmt": text_src(0.03, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic math)
            "cst": text_src(0.15),  # HuggingFaceTB/cosmopedia (synthetic stories)
            "uch": chat_to_sft_src(0.05, input_field="messages", min_chars=200),  # HuggingFaceH4/ultrachat_200k (synthetic chat)
        },
    ),
]
