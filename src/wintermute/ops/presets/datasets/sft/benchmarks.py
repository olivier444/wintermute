from __future__ import annotations

from ...model import DatasetPreset


BENCHMARK_SFT_DATASET_PRESETS = [
    # Dataset: bAbI QA (`facebook/babi_qa`, en-10k-qa1; provider: Meta AI).
    # Summary: Synthetic stories paired with questions that require one or more supporting facts.
    # Natural format: SFT (structured question answering).
    # Orientation: symbolic reasoning, reading comprehension.
    # Recommended SFT task spec: `information_extraction` (constant).
    # Volume: very small (about 2,200 training examples for this task).
    DatasetPreset(
        uid="fbqa",
        name="babi_qa",
        dataset_name="facebook/babi_qa",
        text_fields=("story",),
        get_raw_output_max_shards=200,
        config_template="en-10k-qa1",
    ),
    # Dataset: CommonsenseQA (`tau/commonsense_qa`; provider: Tel Aviv University NLP).
    # Summary: Multiple-choice questions designed to require everyday commonsense knowledge.
    # Natural format: SFT (multiple-choice QA).
    # Orientation: commonsense reasoning.
    # Recommended SFT task spec: `knowledge_qa` (constant).
    # Volume: small (about 12,000 training examples).
    DatasetPreset(
        uid="tcs",
        name="commonsense_qa",
        dataset_name="tau/commonsense_qa",
        text_fields=("question", "choices", "answerKey"),
        get_raw_output_max_shards=200,
        get_raw_stat_fields=["question_concept"],
    ),
    # Dataset: WinoGrande (`allenai/winogrande`, winogrande_debiased; provider: Ai2).
    # Summary: Fill-in-the-blank problems that resolve ambiguous pronoun references.
    # Natural format: SFT (binary multiple choice).
    # Orientation: commonsense reasoning, coreference.
    # Recommended SFT task spec: `problem_solving` (constant).
    # Volume: medium (about 40,000 training examples).
    DatasetPreset(
        uid="wgr",
        name="winogrande",
        dataset_name="allenai/winogrande",
        text_fields=("sentence", "option1", "option2", "answer"),
        get_raw_output_max_shards=200,
        config_template="winogrande_debiased",
    ),
    # Dataset: BoolQ (`google/boolq`; provider: Google Research).
    # Summary: Naturally occurring yes/no questions paired with Wikipedia passages.
    # Natural format: SFT (binary QA).
    # Orientation: reading comprehension, factual reasoning.
    # Recommended SFT task spec: `information_extraction` (constant).
    # Volume: small (about 9,400 training examples).
    DatasetPreset(
        uid="gbq",
        name="boolq",
        dataset_name="google/boolq",
        text_fields=("passage", "question", "answer"),
        get_raw_output_max_shards=200,
    ),
    # Dataset: AI2 ARC (`allenai/ai2_arc`, ARC-Easy; provider: Ai2).
    # Summary: Grade-school science questions with four answer choices.
    # Natural format: SFT (multiple-choice QA).
    # Orientation: science reasoning, school knowledge.
    # Recommended SFT task spec: `knowledge_qa` (constant).
    # Volume: small (about 2,250 training examples in ARC-Easy).
    DatasetPreset(
        uid="aia",
        name="ai2_arc",
        dataset_name="allenai/ai2_arc",
        text_fields=("question", "choices", "answerKey"),
        get_raw_output_max_shards=200,
        config_template="ARC-Easy",
    ),
    # Dataset: PIQA (`ybisk/piqa`; provider: PIQA authors).
    # Summary: Physical commonsense problems with two candidate solutions.
    # Natural format: SFT (binary multiple choice).
    # Orientation: physical commonsense reasoning.
    # Recommended SFT task spec: `knowledge_qa` (constant).
    # Volume: small (about 16,000 training examples).
    DatasetPreset(
        uid="piq",
        name="piqa",
        dataset_name="ybisk/piqa",
        text_fields=("goal", "sol1", "sol2", "label"),
        get_raw_output_max_shards=200,
        config_template="plain_text",
        get_raw_streaming=False,
    ),
    # Dataset: GSM8K (`openai/gsm8k`; provider: OpenAI).
    # Summary: Grade-school word problems with natural-language multi-step solutions.
    # Natural format: SFT with chain-of-thought.
    # Orientation: mathematical reasoning.
    # Recommended SFT task spec: `problem_solving` (constant).
    # Volume: small (7,473 training examples).
    DatasetPreset(
        uid="g8k",
        name="gsm8k",
        dataset_name="openai/gsm8k",
        text_fields=("question", "answer"),
        get_raw_output_max_shards=200,
        config_template="main",
    ),
    # Dataset: MAWPS-ASDiv-SVAMP (`cq01/mawps-asdiv-a_svamp`; provider: CQ01).
    # Summary: Arithmetic word problems represented by templates, operands, equations, and answers.
    # Natural format: Source records for runtime mathematical QA augmentation and SFT rendering.
    # Orientation: mathematical reasoning, arithmetic.
    # Recommended SFT task spec: `problem_solving` (constant).
    # Volume: 3,138 source problems; runtime views emit the original plus up to 100 numeric variants.
    DatasetPreset(
        uid="cqm-raw",
        name="asdiv-a_svamp-raw",
        dataset_name="cq01/mawps-asdiv-a_svamp",
        text_fields=("Question", "Numbers", "Equation", "Answer", "group_nums"),
        get_raw_output_max_shards=200,
        config_template="default",
    ),
    # Dataset: MAWPS (`garrethlee/MAWPS`; provider: MAWPS, republished by garrethlee).
    # Summary: Mathematical word problems paired with equation-form reasoning and numeric answers.
    # Natural format: Source records for runtime mathematical QA augmentation and SFT rendering.
    # Orientation: mathematical reasoning, arithmetic.
    # Recommended SFT task spec: `problem_solving` (constant).
    # Volume: 1,417 source problems; runtime views emit the original plus up to 100 numeric variants.
    DatasetPreset(
        uid="mawps-raw",
        name="MAWPS-raw",
        dataset_name="garrethlee/MAWPS",
        text_fields=("question", "answer"),
        get_raw_output_max_shards=200,
        config_template="default",
    ),
]
