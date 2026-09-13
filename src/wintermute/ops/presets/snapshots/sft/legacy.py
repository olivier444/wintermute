from __future__ import annotations

from wintermute.data.constants import (
    FLD_GENERIC_COMPLETION,
    FLD_GENERIC_PROMPT,
    TASK_CLASSIFICATION,
    TASK_GENERATION,
    TASK_INFORMATION_EXTRACTION,
    TASK_KNOWLEDGE_QA,
    TASK_PROBLEM_SOLVING,
    TASK_REWRITING,
    TASK_SUMMARIZATION,
)

from ..builders import chat_to_sft_src, field_mapping_src, to_sft_src

from ...model import DedupConfig, SnapshotPreset

_SMOLTALK_TASK_SPEC = {
    "field": "source",
    "mapping": {
        "smol-summarize-20k": TASK_SUMMARIZATION,
        "smollm-rewrite-30k": TASK_REWRITING,
        "explore-instruct-rewrite": TASK_REWRITING,
        "smol-summarize-5k": None,
        "smol-magpie-ultra-short": None,
        "self-oss-instruct": TASK_GENERATION,
        "smol-contraints": None,
        "openhermes-50k": None,
        "everyday-conversations": None,
        "longalign": None,
    },
}

_MAGPIE_LLAMA_TASK_SPEC = {
    "field": "task_category",
    "mapping": {
        "Information seeking": TASK_KNOWLEDGE_QA,
        "Math": TASK_PROBLEM_SOLVING,
        "Reasoning": TASK_PROBLEM_SOLVING,
        "Data analysis": TASK_PROBLEM_SOLVING,
        "Editing": None,
        "Advice seeking": TASK_GENERATION,
        "Coding & Debugging": None,
        "Role playing": TASK_GENERATION,
        "Planning": TASK_GENERATION,
        "Brainstorming": TASK_GENERATION,
        "Creative writing": TASK_GENERATION,
    },
}

_NO_ROBOTS_TASK_SPEC = {
    "field": "category",
    "mapping": {
        "Summarize": TASK_SUMMARIZATION,
        "Open QA": TASK_KNOWLEDGE_QA,
        "Closed QA": TASK_INFORMATION_EXTRACTION,
        "Extract": TASK_INFORMATION_EXTRACTION,
        "Classify": TASK_CLASSIFICATION,
        "Generation": TASK_GENERATION,
        "Rewrite": TASK_REWRITING,
        "Brainstorm": TASK_GENERATION,
        "Coding": TASK_GENERATION,
        "Chat": None,
    },
}

_OPEN_PLATYPUS_TASK_SPEC = {
    "field": "data_source",
    "mapping": {
        "MATH/PRM-800K": TASK_PROBLEM_SOLVING,
        "ARB": TASK_PROBLEM_SOLVING,
        "scibench": TASK_PROBLEM_SOLVING,
        "theoremqa": TASK_PROBLEM_SOLVING,
        "reclor": TASK_PROBLEM_SOLVING,
        "scienceqa": TASK_KNOWLEDGE_QA,
        "leetcode_ne": TASK_GENERATION,
        "airoboros": None,
        "guanaco": None,
        "tigerbot-kaggle": None,
    },
}

_DOLLY_TASK_SPEC = {
    "field": "category",
    "mapping": {
        "closed_qa": TASK_INFORMATION_EXTRACTION,
        "open_qa": TASK_KNOWLEDGE_QA,
        "general_qa": TASK_KNOWLEDGE_QA,
        "information_extraction": TASK_INFORMATION_EXTRACTION,
        "classification": TASK_CLASSIFICATION,
        "summarization": TASK_SUMMARIZATION,
        "brainstorming": TASK_GENERATION,
        "creative_writing": TASK_GENERATION,
    },
}

LEGACY_SFT_SNAPSHOT_PRESETS = [
    # Snapshot: Default Assistant SFT v1 (`default1-sft`).
    # Purpose: First general-purpose assistant fine-tuning recipe.
    # Output format: Multi-turn SFT prompt/completion records.
    # Composition: SmolTalk, UltraChat, Open-Platypus, OASST1, No Robots, and Dolly.
    # Task mapping: Complete metadata mappings for SmolTalk, Open-Platypus, No Robots, and Dolly;
    # unreliable known values use explicit None. UltraChat and OASST1 remain unlabeled.
    # Budget: 560M train chars; 0.5M eval chars.
    # Selection: Deduplicated weighted mixture with oversampling of compact instruction sources.
    SnapshotPreset(
        name="default1-sft",
        snapshot_id="default1-sft",
        split_sizes_chars={
            "train": int(140e6 * 4),
            "eval": int(500e3),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=14,
        source_configs={
            "smt": chat_to_sft_src(  # HuggingFaceTB/smol-smoltalk (assistant chat)
                0.35, input_field="messages",
                oversampling=1,
                task=_SMOLTALK_TASK_SPEC,
            ),
            "uch": chat_to_sft_src(  # HuggingFaceH4/ultrachat_200k (synthetic chat)
                0.2, input_field="messages",
                oversampling=1,
            ),
            "opy": field_mapping_src(  # garage-bAInd/Open-Platypus (STEM instruction)
                0.15,
                oversampling=3,
                min_chars=1,
                fields={
                    FLD_GENERIC_PROMPT: ["instruction", "input"],
                    FLD_GENERIC_COMPLETION: ["output"],
                },
                task=_OPEN_PLATYPUS_TASK_SPEC,
            ),
            "oa1": chat_to_sft_src(  # OpenAssistant/oasst1 (human assistant chat)
                0.15, input_field="messages",
                oversampling=3,
                content_field="text",
                user_role="prompter",
            ),
            "nor": chat_to_sft_src(  # HuggingFaceH4/no_robots (human instruction)
                0.05, input_field="messages",
                oversampling=3,
                task=_NO_ROBOTS_TASK_SPEC,
            ),
            "d15": field_mapping_src(  # databricks/databricks-dolly-15k (human instruction)
                0.05,
                oversampling=3,
                min_chars=1,
                fields={
                    FLD_GENERIC_PROMPT: ["instruction", "context"],
                    FLD_GENERIC_COMPLETION: ["response"],
                },
                task=_DOLLY_TASK_SPEC,
            ),
        },
    ),    
    # Snapshot: Default Assistant SFT v2 (`default2-sft`).
    # Purpose: Broader general-purpose assistant fine-tuning recipe.
    # Output format: Multi-turn SFT prompt/completion records.
    # Composition: Assistant conversations, Magpie, SlimOrca, No Robots, OASST1, Open-Platypus, and Dolly.
    # Task mapping: Complete metadata mappings for SmolTalk, Magpie Llama, No Robots,
    # Open-Platypus, and Dolly. Magpie applies its record-level label to every assistant turn;
    # UltraChat, Magpie Pro, SlimOrca, Everyday Conversations, and OASST1 remain unlabeled.
    # Budget: 680M train chars; 20M eval chars.
    # Selection: Deduplicated weighted mixture with selective oversampling of smaller conversation sources.
    SnapshotPreset(
        name="default2-sft",
        snapshot_id="default2-sft",
        split_sizes_chars={
            "train": int(170e6 * 4),
            "eval": int(20e6),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=18,
        source_configs={
            "smt": chat_to_sft_src(  # HuggingFaceTB/smol-smoltalk (assistant chat)
                weight=0.37, input_field="messages",
                oversampling=1,
                task=_SMOLTALK_TASK_SPEC,
            ),
            "uch": chat_to_sft_src(  # HuggingFaceH4/ultrachat_200k (synthetic chat)
                weight=0.1, input_field="messages", #0.1/0.15 (0.2 init)
                oversampling=1,
            ),
            "mlp3": chat_to_sft_src(  # Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered (synthetic instruction)
                weight=0.06, input_field="conversations",
                oversampling=1,
                role_field="from",
                content_field="value",
                user_role="human",
                assistant_role="gpt",
                task=_MAGPIE_LLAMA_TASK_SPEC,
            ),
            "mpm": chat_to_sft_src(  # Magpie-Align/Magpie-Pro-300K-Filtered (synthetic instruction)
                weight=0.06, input_field="conversations",
                oversampling=1,
                role_field="from",
                content_field="value",
                user_role="human",
                assistant_role="gpt",
            ),
            "sor": chat_to_sft_src(  # Open-Orca/SlimOrca (instruction chat)
                weight=0.1, input_field="conversations",
                oversampling=1,
                role_field="from",
                content_field="value",
                user_role="human",
                assistant_role="gpt",
            ),
            "nor": chat_to_sft_src(  # HuggingFaceH4/no_robots (human instruction)
                weight=0.03, input_field="messages",
                oversampling=2,
                task=_NO_ROBOTS_TASK_SPEC,
            ),
            "edc": chat_to_sft_src(  # HuggingFaceTB/everyday-conversations-llama3.1-2k (casual chat)
                weight=0.01, input_field="messages",
                oversampling=4,
            ),
            "oa1": chat_to_sft_src(  # OpenAssistant/oasst1 (human assistant chat)
                weight=0.08, input_field="messages",
                oversampling=2,
                content_field="text",
                user_role="prompter",
            ),
            "opy": field_mapping_src(  # garage-bAInd/Open-Platypus (STEM instruction)
                weight=0.035,
                oversampling=1,
                min_chars=1,
                fields={
                    FLD_GENERIC_PROMPT: ["instruction", "input"],
                    FLD_GENERIC_COMPLETION: ["output"],
                },
                task=_OPEN_PLATYPUS_TASK_SPEC,
            ),
            "d15": field_mapping_src(  # databricks/databricks-dolly-15k (human instruction)
                weight=0.01,
                oversampling=1,
                min_chars=1,
                fields={
                    FLD_GENERIC_PROMPT: ["instruction", "context"],
                    FLD_GENERIC_COMPLETION: ["response"],
                },
                task=_DOLLY_TASK_SPEC,
            ),
        },
    ),
    # Snapshot: Reasoning Benchmarks SFT v1 (`abstraction1-sft`).
    # Purpose: Build general reasoning skills from structured QA and math tasks.
    # Output format: SFT prompt/completion records, with explicit reasoning where available.
    # Composition: bAbI, commonsense, reading, science, physical reasoning, and GSM8K.
    # Task mapping: extraction, knowledge QA, and problem-solving labels follow each source behavior.
    # Budget: 35M train chars; 2M eval chars.
    # Selection: Deduplicated weighted mixture with source-specific renderers and oversampling.
    SnapshotPreset(
        name="abstraction1-sft",
        snapshot_id="abstraction1-sft",
        split_sizes_chars={
            "train": int(35e6),
            "eval": int(2e6),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=29,
        source_configs={
            "fbqa": to_sft_src(  # facebook/babi_qa (symbolic QA)
                weight=0.85,
                oversampling=3,
                items_field="story",
                context=[{"field": "text"}],
                context_filter={"field": "type", "equals": 0},
                prompt=[{"text": "Question:"}, {"field": "text"}, {"text": "Answer with a short fact."}],
                completion=[{"field": "answer"}],
                emit_filter={"field": "type", "equals": 1},
                context_joiner="\n",
                task=TASK_INFORMATION_EXTRACTION,
            ),
            "tcs": to_sft_src(  # tau/commonsense_qa (commonsense QA)
                weight=1.0,
                oversampling=3,
                prompt=[
                    {"text": "Question:"},
                    {"field": "question"},
                    {"text": "Choices:"},
                    {"field": "choices", "transform": "choices"},
                    {"text": "Answer with the best option text."},
                ],
                completion=[{"field": "answerKey", "transform": "choice_text_by_label", "choices_field": "choices"}],
                task=TASK_KNOWLEDGE_QA,
            ),
            "wgr": to_sft_src(  # allenai/winogrande (coreference QA)
                weight=0.62,
                oversampling=3,
                prompt=[
                    {"text": "Fill in the blank with the best option."},
                    {"text": "Sentence:"},
                    {"field": "sentence"},
                    {"text": "Options:"},
                    {"fields": ["option1", "option2"], "transform": "numbered_choices"},
                ],
                completion=[
                    {
                        "field": "answer",
                        "transform": "choice_text_by_index",
                        "choices_fields": ["option1", "option2"],
                        "one_based": True,
                    }
                ],
                task=TASK_PROBLEM_SOLVING,
            ),
            "gbq": to_sft_src(  # google/boolq (boolean QA)
                weight=1,
                oversampling=1,
                prompt=[
                    {"text": "Passage:"},
                    {"field": "passage"},
                    {"text": "Question:"},
                    {"field": "question"},
                    {"text": "Answer with yes or no."},
                ],
                completion=[{"field": "answer", "transform": "bool_yes_no"}],
                task=TASK_INFORMATION_EXTRACTION,
            ),
            "aia": to_sft_src(  # allenai/ai2_arc (science QA)
                weight=0.32,
                oversampling=3,
                prompt=[
                    {"text": "Question:"},
                    {"field": "question"},
                    {"text": "Choices:"},
                    {"field": "choices", "transform": "choices"},
                    {"text": "Answer with the best option text."},
                ],
                completion=[{"field": "answerKey", "transform": "choice_text_by_label", "choices_field": "choices"}],
                task=TASK_KNOWLEDGE_QA,
            ),
            "piq": to_sft_src(  # ybisk/piqa (physical commonsense)
                weight=0.7,
                oversampling=1,
                prompt=[
                    {"text": "Goal:"},
                    {"field": "goal"},
                    {"text": "Choices:"},
                    {"fields": ["sol1", "sol2"], "transform": "numbered_choices"},
                    {"text": "Answer with the best option text."},
                ],
                completion=[
                    {
                        "field": "label",
                        "transform": "choice_text_by_index",
                        "choices_fields": ["sol1", "sol2"],
                        "one_based": False,
                    }
                ],
                task=TASK_KNOWLEDGE_QA,
            ),
            "g8k": to_sft_src(  # openai/gsm8k (math reasoning)
                weight=0.73,
                oversampling=1,
                prompt=[
                    {"text": "Solve the math problem. Show the reasoning, then give the final answer."},
                    {"field": "question"},
                ],
                completion=[{"field": "answer", "transform": "math_reasoning"}],
                task=TASK_PROBLEM_SOLVING,
            ),
        },
    ),
]
