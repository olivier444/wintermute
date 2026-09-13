from __future__ import annotations

from ..builders import xp3_dataset
from ...model import DatasetPreset
from .xp3_sources import XP3_EN_GENERAL_URLS_BY_TASK

GENERAL_KNOWLEDGE_SFT_DATASET_PRESETS = [
    # Dataset: xP3 English General Classification (`bigscience/xP3`; provider: BigScience).
    # Summary: English WikiQA answer-validation and relevance classification templates.
    # Natural format: SFT (prompt to target).
    # Orientation: classification, English.
    # Recommended SFT task spec: `classification` (constant).
    # Volume: five selected English xP3 task templates.
    xp3_dataset(
        uid="xp3-en-gk-class",
        name="xp3-en-general-classification",
        language="en",
        data_files=XP3_EN_GENERAL_URLS_BY_TASK["classification"],
        output_max_shards=100,
    ),
    # Dataset: xP3 English General Generation (`bigscience/xP3`; provider: BigScience).
    # Summary: English question, Jeopardy-style clue, and process-continuation generation templates.
    # Natural format: SFT (prompt to target).
    # Orientation: generation, English.
    # Recommended SFT task spec: `generation` (constant).
    # Volume: four selected English xP3 task templates.
    xp3_dataset(
        uid="xp3-en-gk-gen",
        name="xp3-en-general-generation",
        language="en",
        data_files=XP3_EN_GENERAL_URLS_BY_TASK["generation"],
        output_max_shards=100,
    ),
    # Dataset: xP3 English Information Extraction (`bigscience/xP3`; provider: BigScience).
    # Summary: English WikiQA topic extraction from supplied questions and answers.
    # Natural format: SFT (prompt to target).
    # Orientation: information extraction, English.
    # Recommended SFT task spec: `information_extraction` (constant).
    # Volume: three selected English xP3 task templates.
    xp3_dataset(
        uid="xp3-en-ie",
        name="xp3-en-information-extraction",
        language="en",
        data_files=XP3_EN_GENERAL_URLS_BY_TASK["information_extraction"],
        output_max_shards=100,
    ),
    # Dataset: xP3 English Knowledge QA (`bigscience/xP3`; provider: BigScience).
    # Summary: English closed-book TriviaQA, WebQuestions, and WikiQA answer templates.
    # Natural format: SFT (prompt to target).
    # Orientation: knowledge question answering, English.
    # Recommended SFT task spec: `knowledge_qa` (constant).
    # Volume: ten selected English xP3 task templates.
    xp3_dataset(
        uid="xp3-en-kqa",
        name="xp3-en-knowledge-qa",
        language="en",
        data_files=XP3_EN_GENERAL_URLS_BY_TASK["knowledge_qa"],
        output_max_shards=200,
    ),
    # Dataset: xP3 English Problem Solving (`bigscience/xP3`; provider: BigScience).
    # Summary: English ARC-Challenge and WIQA scientific or procedural inference templates.
    # Natural format: SFT (prompt to target).
    # Orientation: problem solving, English.
    # Recommended SFT task spec: `problem_solving` (constant).
    # Volume: fourteen selected English xP3 task templates.
    xp3_dataset(
        uid="xp3-en-ps",
        name="xp3-en-problem-solving",
        language="en",
        data_files=XP3_EN_GENERAL_URLS_BY_TASK["problem_solving"],
        output_max_shards=200,
    ),
    # Dataset: Aya (`CohereLabs/aya_dataset`; provider: Cohere For AI).
    # Summary: Multilingual instruction-response pairs, filtered here to French.
    # Natural format: SFT (instruction to response).
    # Orientation: general knowledge, French.
    # Recommended SFT task spec: omitted; `annotation_type` describes provenance, not task intent.
    # Volume: medium (about 200,000 examples across languages).
    DatasetPreset(
        uid="aya-fr",
        name="aya-fr",
        dataset_name="CohereLabs/aya_dataset",
        text_fields=("inputs", "targets"),
        get_raw_output_max_shards=200,
        config_template="default",
        lan_field="language_code",
        lang=["fra"],
        get_raw_stat_fields=["annotation_type"],
        get_raw_exclude_fields=["user_id"],
    ),
]
