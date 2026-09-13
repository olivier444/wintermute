from __future__ import annotations

from ..builders import chat_to_sft_src, text_src
from ...model import DedupConfig, SnapshotPreset

DEFAULT_TOKENIZER_SPLIT_SIZES = {
    "small": int(600e6),
    "medium": int(1200e6),
    "big": int(2500e6),
}

DEFAULT_TOKENIZER_SNAPSHOT_PRESETS = [
    # Snapshot: Default Tokenizer Sample (`default-tok`).
    # Purpose: Train tokenizer candidates on a broad English text mixture.
    # Output format: Text plus structured prompt/completion records.
    # Composition: Web, books, synthetic education, and UltraChat.
    # Budget: 600M, 1.2B, or 2.5B chars for the small, medium, or big split.
    # Selection: Deduplicated weighted mixture with source-specific length limits.
    SnapshotPreset(
        name="default-tok",
        snapshot_id="default-tok",
        split_sizes_chars=dict(DEFAULT_TOKENIZER_SPLIT_SIZES),
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=342,
        source_configs={
            "wk": text_src(0.15),  # wikimedia/wikipedia (English encyclopedia)
            "fwe-2": text_src(0.3),  # HuggingFaceFW/fineweb-edu (educational web)
            "fwe": text_src(0.2),  # HuggingFaceFW/fineweb-edu (educational web)
            "pg": text_src(0.05, min_chars=800),  # emozilla/pg19 (public-domain books)
            "ckh": text_src(0.05, oversampling=3, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic educational text)
            "cox": text_src(0.05, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic textbooks)
            "cwh": text_src(0.1, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic procedural text)
            "cmt": text_src(0.05, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic math)
            "cst": text_src(0.1),  # HuggingFaceTB/cosmopedia (synthetic stories)
            "uch": chat_to_sft_src(0.05, input_field="messages", min_chars=200),  # HuggingFaceH4/ultrachat_200k (synthetic chat)
        },
    ),
    # Snapshot: Multilingual Tokenizer Sample (`mix-tok`).
    # Purpose: Train a tokenizer covering English/French text, code, mathematics, and reasoning.
    # Output format: Text plus structured prompt/completion records.
    # Composition: Web, books, synthetic education, code, mathematics, and OpenThoughts.
    # Budget: 2.5B chars in the big split.
    # Selection: Deduplicated weighted mixture with source-specific length limits.
    SnapshotPreset(
        name="mix-tok",
        snapshot_id="mix-tok",
        split_sizes_chars={
            "big": int(2.5e9),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=2041,
        source_configs={
            "fwe-2": text_src(0.30, oversampling=1),  # HuggingFaceFW/fineweb-edu (educational web)
            "fwe": text_src(0.07, oversampling=1),  # HuggingFaceFW/fineweb-edu (educational web)
            "wk": text_src(0.10, oversampling=1),  # wikimedia/wikipedia (English encyclopedia)
            "pg": text_src(0.05, oversampling=1, min_chars=1200),  # emozilla/pg19 (public-domain books)
            "cst": text_src(0.03, oversampling=1, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic stories)
            "cox": text_src(0.04, oversampling=1, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic textbooks)
            "cwh": text_src(0.02, oversampling=1, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic procedural text)
            "ckh": text_src(0.01, oversampling=1, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic educational text)
            "fw2-fr": text_src(0.15, oversampling=1),  # epfml/FineWeb2-HQ (high-quality French web)
            "wk-fr": text_src(0.05, oversampling=1),  # wikimedia/wikipedia (French encyclopedia)
            "stp2": text_src(0.08, field="content", oversampling=1, min_chars=200),  # bigcode/the-stack-dedup (Python code)
            "sto2": text_src(0.04, field="content", oversampling=1, min_chars=200),  # bigcode/the-stack-dedup (multi-language code)
            "owm": text_src(0.04, oversampling=1, min_chars=400),  # open-web-math/open-web-math (math web)
            "cmt": text_src(0.015, oversampling=1, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic math)
            "oth": chat_to_sft_src(  # open-thoughts/OpenThoughts3-1.2M (reasoning chat)
                0.005, input_field="conversations",
                oversampling=1,
                min_chars=200,
                role_field="from",
                content_field="value",
                user_role="human",
                assistant_role="gpt",
                assistant_content_regex=r"\s*<think>\s*(.*?)\s*</think>\s*(.*)\s*\Z",
            ),
        },
    ),
]
