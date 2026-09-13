from __future__ import annotations

from wintermute.data.constants import FLD_GENERIC_TEXT

from ..builders import chat_to_sft_src, field_mapping_src, text_src

from ...model import DedupConfig, SnapshotPreset

DEFAULT5_TXT_SNAPSHOT_ID = "default5-txt"

CORE_PRETRAIN_SNAPSHOT_PRESETS = [
    # Snapshot: Default Text (`default5-txt`).
    # Purpose: Primary broad pretraining recipe.
    # Output format: NTP text plus structured prompt/completion records.
    # Composition: English web, mathematics, books, synthetic education, code, and a small chat dose.
    # Budget: 56B train chars; 10M eval chars; 50M test chars.
    # Selection: Deduplicated weighted mixture with source-specific length limits.
    SnapshotPreset(
        name="default-txt",
        snapshot_id="default5-txt",
        split_sizes_chars={
            "train": int(14e9 * 4), # 14 bn tokens
            "eval": int(1e7),
            "test": int(5e7),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=1774,
        source_configs={
            "fwe-2": text_src(0.19, oversampling=1),  # HuggingFaceFW/fineweb-edu (educational web)
            "fwe": text_src(0.15, oversampling=1),  # HuggingFaceFW/fineweb-edu (educational web)
            "wk": text_src(0.11, oversampling=1),  # wikimedia/wikipedia (English encyclopedia)
            "owm": text_src(0.16, oversampling=1, min_chars=400),  # open-web-math/open-web-math (math web)
            "cmt": text_src(0.09, oversampling=1, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic math)
            "cst": text_src(0.09, oversampling=1, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic stories)
            "pg": text_src(0.09, oversampling=1, min_chars=1200),  # emozilla/pg19 (public-domain books)
            "cox": text_src(0.015, oversampling=2, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic textbooks)
            "cwh": text_src(0.025, oversampling=2, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic procedural text)
            "ckh": text_src(0.003, oversampling=2, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic educational text)
            "stp": text_src(0.049, field="content", oversampling=1, min_chars=200),  # bigcode/the-stack-dedup (Python code)
            "sto": text_src(0.018, field="content", oversampling=1, min_chars=200),  # bigcode/the-stack-dedup (multi-language code)
            "uch": chat_to_sft_src(0.01, input_field="messages", min_chars=200),  # HuggingFaceH4/ultrachat_200k (synthetic chat)
        },
    ),
    # Snapshot: Default Continuation Text (`default6-txt`).
    # Purpose: Continue broad pretraining after the Default Text snapshot.
    # Output format: NTP text plus structured prompt/completion records.
    # Composition: The core mixture with adjusted code and synthetic-education weights.
    # Budget: 60B train chars; 10M eval chars; 50M test chars.
    # Selection: Deduplicated; excludes Default Text members for configured overlapping sources.
    SnapshotPreset(
        name="default-continuation-txt",
        snapshot_id="default6-txt",
        split_sizes_chars={
            "train": int(15e9 * 4), # 15 bn tokens
            "eval": int(1e7),
            "test": int(5e7),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=1375,
        source_configs={
            "fwe-2": text_src(0.19, oversampling=1, excluded_snapshot_ids=[DEFAULT5_TXT_SNAPSHOT_ID]),  # HuggingFaceFW/fineweb-edu (educational web)
            "fwe": text_src(0.15, oversampling=1, excluded_snapshot_ids=[DEFAULT5_TXT_SNAPSHOT_ID]),  # HuggingFaceFW/fineweb-edu (educational web)
            "wk": text_src(0.11, oversampling=1, excluded_snapshot_ids=[DEFAULT5_TXT_SNAPSHOT_ID]),  # wikimedia/wikipedia (English encyclopedia)
            "owm": text_src(0.16, oversampling=1, min_chars=400, excluded_snapshot_ids=[DEFAULT5_TXT_SNAPSHOT_ID]),  # open-web-math/open-web-math (math web)
            "cmt": text_src(0.08, oversampling=1, min_chars=800, excluded_snapshot_ids=[DEFAULT5_TXT_SNAPSHOT_ID]),  # HuggingFaceTB/cosmopedia (synthetic math)
            "cst": text_src(0.09, oversampling=1, min_chars=800, excluded_snapshot_ids=[DEFAULT5_TXT_SNAPSHOT_ID]),  # HuggingFaceTB/cosmopedia (synthetic stories)
            "pg": text_src(0.09, oversampling=1, min_chars=1200, excluded_snapshot_ids=[DEFAULT5_TXT_SNAPSHOT_ID]),  # emozilla/pg19 (public-domain books)
            "cox": text_src(0.015, oversampling=2, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic textbooks)
            "cwh": text_src(0.025, oversampling=2, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic procedural text)
            "ckh": text_src(0.003, oversampling=2, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic educational text)
            "stp": text_src(0.1, field="content", oversampling=1, min_chars=200, excluded_snapshot_ids=[DEFAULT5_TXT_SNAPSHOT_ID]),  # bigcode/the-stack-dedup (Python code)
            "sto": text_src(0.02, field="content", oversampling=1, min_chars=200, excluded_snapshot_ids=[DEFAULT5_TXT_SNAPSHOT_ID]),  # bigcode/the-stack-dedup (multi-language code)
            "uch": chat_to_sft_src(0.01, input_field="messages", min_chars=200),  # HuggingFaceH4/ultrachat_200k (synthetic chat)
        },
    ),
    # Snapshot: Mix 40B Text (`mix40b-txt`).
    # Purpose: Large multilingual pretraining mixture with targeted instruction exposure.
    # Output format: NTP text plus structured prompt/completion records.
    # Composition: English/French web, mathematics, synthetic education, code, and structured assistant data.
    # Budget: 160B train chars; 10M eval chars; 50M test chars.
    # Selection: Deduplicated weighted mixture with source-specific oversampling and length limits.
    SnapshotPreset(
        name="mix40b-txt",
        snapshot_id="mix40b-txt",
        split_sizes_chars={
            "train": int(40e9 * 4),
            "eval": int(1e7),
            "test": int(5e7),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=2027,
        source_configs={
            "fwe-2": text_src(12.0, oversampling=1),  # HuggingFaceFW/fineweb-edu (educational web)
            "fwe": text_src(2.4, oversampling=1),  # HuggingFaceFW/fineweb-edu (educational web)
            "wk": text_src(2.2, oversampling=1),  # wikimedia/wikipedia (English encyclopedia)
            "pg": text_src(2.2, oversampling=1, min_chars=1200),  # emozilla/pg19 (public-domain books)
            "cst": text_src(1.0, oversampling=1, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic stories)
            "cox": text_src(0.7, oversampling=6, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic textbooks)
            "cwh": text_src(0.5, oversampling=3, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic procedural text)
            "owm": text_src(5.0, oversampling=1, min_chars=400),  # open-web-math/open-web-math (math web)
            "oth": chat_to_sft_src(  # open-thoughts/OpenThoughts3-1.2M (reasoning chat)
                4.0, input_field="conversations",
                oversampling=1,
                min_chars=200,
                role_field="from",
                content_field="value",
                user_role="human",
                assistant_role="gpt",
                assistant_content_regex=r"\s*<think>\s*(.*?)\s*</think>\s*(.*)\s*\Z",
            ),
            "cmt": text_src(1.4, oversampling=1, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic math)
            "ckh": text_src(0.25, oversampling=12, min_chars=800),  # HuggingFaceTB/cosmopedia (synthetic educational text)
            "stp2": text_src(3.0, field="content", oversampling=1, min_chars=200),  # bigcode/the-stack-dedup (Python code)
            "sto2": text_src(1.0, field="content", oversampling=1, min_chars=200),  # bigcode/the-stack-dedup (multi-language code)
            "fw2-fr": text_src(5.0, oversampling=1),  # epfml/FineWeb2-HQ (high-quality French web)
            "wk-fr": text_src(1.2, oversampling=1),  # wikimedia/wikipedia (French encyclopedia)
            "smt": chat_to_sft_src(0.15, input_field="messages", min_chars=200),  # HuggingFaceTB/smol-smoltalk (assistant chat)
            "uch": chat_to_sft_src(0.1, input_field="messages", min_chars=200),  # HuggingFaceH4/ultrachat_200k (synthetic chat)
            "mlp3": chat_to_sft_src(  # Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered (synthetic instruction)
                0.05, input_field="conversations", min_chars=200,
                role_field="from",
                content_field="value",
                user_role="human",
                assistant_role="gpt",
            ),
            "mpm": chat_to_sft_src(  # Magpie-Align/Magpie-Pro-300K-Filtered (synthetic instruction)
                0.03, input_field="conversations", min_chars=200,
                role_field="from",
                content_field="value",
                user_role="human",
                assistant_role="gpt",
            ),
            "sor": chat_to_sft_src(  # Open-Orca/SlimOrca (instruction chat)
                0.04, input_field="conversations", min_chars=200,
                role_field="from",
                content_field="value",
                user_role="human",
                assistant_role="gpt",
            ),
            "oa1": chat_to_sft_src(  # OpenAssistant/oasst1 (human assistant chat)
                0.007, input_field="messages", min_chars=200,
                content_field="text",
                user_role="prompter",
            ),
            "opy": field_mapping_src(  # garage-bAInd/Open-Platypus (STEM instruction)
                0.015,
                min_chars=1,
                oversampling=2,
                fields={
                    FLD_GENERIC_TEXT: ["instruction", "input", "output"],
                },
            ),
            "d15": field_mapping_src(  # databricks/databricks-dolly-15k (human instruction)
                0.01,
                min_chars=1,
                oversampling=4,
                fields={
                    FLD_GENERIC_TEXT: ["instruction", "context", "response"],
                },
            ),
            "nor": chat_to_sft_src(0.015, input_field="messages", oversampling=5, min_chars=200),  # HuggingFaceH4/no_robots (human instruction)
            "edc": chat_to_sft_src(0.003, input_field="messages", oversampling=6, min_chars=200),  # HuggingFaceTB/everyday-conversations-llama3.1-2k (casual chat)
            "xp3-fr-sum": field_mapping_src(  # bigscience/xP3 (French summarization)
                0.03764198821230372,
                min_chars=1,
                fields={
                    FLD_GENERIC_TEXT: ["inputs", "targets"],
                },
            ),
            "xlwic-fr": field_mapping_src(  # pasinit/xlwic (French word-sense classification)
                0.004292245020334778,
                min_chars=1,
                fields={
                    FLD_GENERIC_TEXT: ["target_word", "context_1", "context_2", "label"],
                },
            ),
            "xp3-fr-gen": field_mapping_src(  # bigscience/xP3 (French generation)
                0.008065766767361504,
                min_chars=1,
                fields={
                    FLD_GENERIC_TEXT: ["inputs", "targets"],
                },
            ),
            "oa2-fr": chat_to_sft_src(  # OpenAssistant/oasst2 (human assistant chat)
                0.004, input_field="messages", min_chars=200,
                content_field="text",
                oversampling=7,
                user_role="prompter",
            ),
            "aya-fr": field_mapping_src(  # CohereLabs/aya_dataset (multilingual instruction)
                0.002,
                min_chars=1,
                oversampling=10,
                fields={
                    FLD_GENERIC_TEXT: ["inputs", "targets"],
                },
            ),
        },
    ),
]
