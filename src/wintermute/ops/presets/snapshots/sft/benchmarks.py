from __future__ import annotations

from .common import unitary_raw_sft_snapshot


BENCHMARK_SFT_SNAPSHOT_PRESETS = [
    # Snapshot: bAbI QA (`babi_qa`).
    # Dataset: bAbI QA (`facebook/babi_qa`, en-10k-qa1; provider: Meta AI).
    # Summary: Synthetic stories paired with questions that require one or more supporting facts.
    # Purpose: Persist the selected raw bAbI QA records; SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `fbqa` raw datasource only.
    # Runtime task mapping: TASK_INFORMATION_EXTRACTION in every benchmark training view.
    # Budget: 1.490M train chars; 74,488 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record once.
    unitary_raw_sft_snapshot(
        "babi_qa",
        source_id="fbqa",
        total_chars=1_564_248,
    ),
    # Snapshot: CommonsenseQA (`commonsense_qa`).
    # Dataset: CommonsenseQA (`tau/commonsense_qa`; provider: Tel Aviv University NLP).
    # Summary: Multiple-choice questions designed to require everyday commonsense knowledge.
    # Purpose: Persist the selected raw CommonsenseQA records; QCM rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `tcs` raw datasource only.
    # Runtime task mapping: TASK_KNOWLEDGE_QA in every QCM training view.
    # Budget: 1.737M train chars; 86,857 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record once.
    unitary_raw_sft_snapshot(
        "commonsense_qa",
        source_id="tcs",
        total_chars=1_824_001,
    ),
    # Snapshot: WinoGrande (`winogrande`).
    # Dataset: WinoGrande (`allenai/winogrande`, winogrande_debiased; provider: Ai2).
    # Summary: Fill-in-the-blank problems that resolve ambiguous pronoun references.
    # Purpose: Persist the selected raw WinoGrande records; QCM rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `wgr` raw datasource only.
    # Runtime task mapping: TASK_PROBLEM_SOLVING in every QCM training view.
    # Budget: 1.058M train chars; 52,899 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record once.
    unitary_raw_sft_snapshot(
        "winogrande",
        source_id="wgr",
        total_chars=1_110_884,
    ),
    # Snapshot: BoolQ (`boolq`).
    # Dataset: BoolQ (`google/boolq`; provider: Google Research).
    # Summary: Naturally occurring yes/no questions paired with Wikipedia passages.
    # Purpose: Persist the selected raw BoolQ records; natural/QCM variants are rendered in the training view.
    # Output format: Raw datasource records.
    # Composition: The `gbq` raw datasource only.
    # Runtime task mapping: TASK_INFORMATION_EXTRACTION in every QCM training view.
    # Budget: 5.548M train chars; 277,415 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record once.
    unitary_raw_sft_snapshot(
        "boolq",
        source_id="gbq",
        total_chars=5_825_715,
    ),
    # Snapshot: AI2 ARC (`ai2_arc`).
    # Dataset: AI2 ARC (`allenai/ai2_arc`, ARC-Easy; provider: Ai2).
    # Summary: Grade-school science questions with four answer choices.
    # Purpose: Persist the selected raw ARC-Easy records; QCM rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `aia` raw datasource only.
    # Runtime task mapping: TASK_KNOWLEDGE_QA in every QCM training view.
    # Budget: 566,143 train chars; 28,307 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record once.
    unitary_raw_sft_snapshot(
        "ai2_arc",
        source_id="aia",
        total_chars=594_450,
    ),
    # Snapshot: PIQA (`piqa`).
    # Dataset: PIQA (`ybisk/piqa`; provider: PIQA authors).
    # Summary: Physical commonsense problems with two candidate solutions.
    # Purpose: Persist the selected raw PIQA records; QCM rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `piq` raw datasource only.
    # Runtime task mapping: TASK_KNOWLEDGE_QA in every QCM training view.
    # Budget: 3.707M train chars; 185,368 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record once.
    unitary_raw_sft_snapshot(
        "piqa",
        source_id="piq",
        total_chars=3_892_745,
    ),
    # Snapshot: GSM8K (`gsm8k`).
    # Dataset: GSM8K (`openai/gsm8k`; provider: OpenAI).
    # Summary: Grade-school word problems with natural-language multi-step solutions.
    # Purpose: Persist the selected raw GSM8K records; reasoning SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `g8k` raw datasource only.
    # Runtime task mapping: TASK_PROBLEM_SOLVING in every benchmark training view.
    # Budget: 3.729M train chars; 186,470 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record once.
    unitary_raw_sft_snapshot(
        "gsm8k",
        source_id="g8k",
        total_chars=3_915_873,
    ),
    # Snapshot: MAWPS-ASDiv-SVAMP (`asdiv-a_svamp-raw`).
    # Dataset: MAWPS-ASDiv-SVAMP (`cq01/mawps-asdiv-a_svamp`; provider: CQ01).
    # Summary: Arithmetic word problems represented by templates, operands, equations, and answers.
    # Purpose: Persist source problems before runtime numeric augmentation and SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `cqm-raw` raw datasource only.
    # Runtime task mapping: TASK_PROBLEM_SOLVING in every benchmark training view.
    # Budget: 764,315 train chars; 38,215 eval chars; together they cover all raw chars.
    # Selection: Every indexed source problem exactly once before augmentation.
    unitary_raw_sft_snapshot(
        "asdiv-a_svamp-raw",
        source_id="cqm-raw",
        total_chars=802_530,
    ),
    # Snapshot: MAWPS (`MAWPS-raw`).
    # Dataset: MAWPS (`garrethlee/MAWPS`; provider: MAWPS, republished by garrethlee).
    # Summary: Mathematical word problems paired with equation-form reasoning and numeric answers.
    # Purpose: Persist source problems before runtime numeric augmentation and SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `mawps-raw` raw datasource only.
    # Runtime task mapping: TASK_PROBLEM_SOLVING in every benchmark training view.
    # Budget: 222,613 train chars; 11,130 eval chars; together they cover all raw chars.
    # Selection: Every indexed source problem exactly once before augmentation.
    unitary_raw_sft_snapshot(
        "MAWPS-raw",
        source_id="mawps-raw",
        total_chars=233_743,
    ),
]
