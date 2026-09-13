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
from wintermute.data.transform import RecordTransformConfig
from wintermute.data.transform.config_builders import chat_to_sft, field_mapping


def ultrachat_transform() -> RecordTransformConfig:
    return chat_to_sft(input_field="messages")


def smoltalk_transform() -> RecordTransformConfig:
    return chat_to_sft(
        input_field="messages",
        task={
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
        },
    )


def magpie_mono_pro_transform() -> RecordTransformConfig:
    return chat_to_sft(
        input_field="conversations",
        role_field="from",
        content_field="value",
        user_role="human",
        assistant_role="gpt",
    )


def magpie_llama_pro_transform() -> RecordTransformConfig:
    return chat_to_sft(
        input_field="conversations",
        role_field="from",
        content_field="value",
        user_role="human",
        assistant_role="gpt",
        task={
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
        },
    )


def slim_orca_transform() -> RecordTransformConfig:
    return chat_to_sft(
        input_field="conversations",
        role_field="from",
        content_field="value",
        user_role="human",
        assistant_role="gpt",
    )


def everyday_conversations_transform() -> RecordTransformConfig:
    return chat_to_sft(input_field="messages")


def no_robots_transform() -> RecordTransformConfig:
    return chat_to_sft(
        input_field="messages",
        task={
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
        },
    )


def open_platypus_transform() -> RecordTransformConfig:
    return field_mapping(
        fields={
            FLD_GENERIC_PROMPT: ["instruction", "input"],
            FLD_GENERIC_COMPLETION: ["output"],
        },
        task={
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
        },
    )


def dolly_15k_transform() -> RecordTransformConfig:
    return field_mapping(
        fields={
            FLD_GENERIC_PROMPT: ["instruction", "context"],
            FLD_GENERIC_COMPLETION: ["response"],
        },
        task={
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
        },
    )


def oasst1_transform() -> RecordTransformConfig:
    return chat_to_sft(
        input_field="messages",
        content_field="text",
        user_role="prompter",
    )


def oasst2_fr_transform() -> RecordTransformConfig:
    return chat_to_sft(
        input_field="messages",
        content_field="text",
        user_role="prompter",
    )


def aya_fr_transform() -> RecordTransformConfig:
    return field_mapping(
        fields={
            FLD_GENERIC_PROMPT: ["inputs"],
            FLD_GENERIC_COMPLETION: ["targets"],
        },
    )
