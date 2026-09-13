from __future__ import annotations

from .common import unitary_raw_sft_snapshot


TEXT_PROCESSING_SFT_SNAPSHOT_PRESETS = [
    # Snapshot: xP3 English Summarization (`xp3-en-summary`).
    # Dataset: xP3 English Summarization (`bigscience/xP3`; provider: BigScience).
    # Summary: English article-condensation instructions rendered through five xP3 templates.
    # Purpose: Persist raw xP3 summarization records for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `xp3-en-sum` raw datasource only.
    # Runtime task mapping: TASK_SUMMARIZATION in every task-facing view.
    # Budget: 878.326M train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "xp3-en-summary",
        source_id="xp3-en-sum",
        total_chars=888_326_374,
        seed=2052,
    ),
    # Snapshot: xP3 English Question Answering (`xp3-en-qa`).
    # Dataset: xP3 English Question Answering (`bigscience/xP3`; provider: BigScience).
    # Summary: English extractive and passage-grounded question answering.
    # Purpose: Persist raw xP3 question-answering records for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `xp3-en-qa` raw datasource only.
    # Runtime task mapping: TASK_INFORMATION_EXTRACTION in every task-facing view.
    # Budget: 164.202M train chars; 8.210M eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "xp3-en-qa",
        source_id="xp3-en-qa",
        total_chars=172_412_251,
        seed=2052,
    ),
    # Snapshot: AG News (`ag_news`).
    # Dataset: AG News (`fancyzhx/ag_news`, default; provider: Xiang Zhang).
    # Summary: English news articles classified into World, Sports, Business, or Science and Technology.
    # Purpose: Persist the selected raw articles; shuffled QCM rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `ag-news` raw datasource only.
    # Runtime task mapping: TASK_CLASSIFICATION in every QCM training view.
    # Budget: 27.369M train chars; 1.368M eval chars; together they cover all raw chars.
    # Selection: Every indexed source record once.
    unitary_raw_sft_snapshot(
        "ag_news",
        source_id="ag-news",
        total_chars=28_737_303,
        seed=2052,
    ),
    # Snapshot: PAWS (`paws`).
    # Dataset: PAWS (`google-research-datasets/paws`, labeled_final; provider: Google Research).
    # Summary: English sentence pairs labelled by whether they are paraphrases.
    # Purpose: Persist the selected raw pairs; natural/QCM variants are rendered in the training view.
    # Output format: Raw datasource records.
    # Composition: The `paws` raw datasource only.
    # Runtime task mapping: TASK_CLASSIFICATION in every QCM training view.
    # Budget: 10.940M train chars; 546,978 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record once.
    unitary_raw_sft_snapshot(
        "paws",
        source_id="paws",
        total_chars=11_486_558,
        seed=2052,
    ),
    # Snapshot: SuperGLUE WiC (`wic`).
    # Dataset: SuperGLUE WiC (`aps/super_glue`, wic; provider: SuperGLUE).
    # Summary: English sentence pairs labelled by whether a target word has the same sense in both.
    # Purpose: Persist the selected raw pairs; natural/QCM variants are rendered in the training view.
    # Output format: Raw datasource records.
    # Composition: The `wic` raw datasource only.
    # Runtime task mapping: TASK_CLASSIFICATION in every QCM training view.
    # Budget: 456,421 train chars; 22,821 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record once.
    unitary_raw_sft_snapshot(
        "wic",
        source_id="wic",
        total_chars=479_242,
        seed=2052,
    ),
    # Snapshot: xP3 English PAWS Rewriting (`xp3-en-paws-rewrite`).
    # Dataset: xP3 English PAWS Rewriting (`bigscience/xP3`; provider: BigScience).
    # Summary: English sentence rewriting from the PAWS paraphrase-generation template.
    # Purpose: Persist raw xP3 paraphrase records for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `xp3-en-paws-rewrite` raw datasource only.
    # Runtime task mapping: TASK_REWRITING in every task-facing view.
    # Budget: 5.283M train chars; 264.158K eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "xp3-en-paws-rewrite",
        source_id="xp3-en-paws-rewrite",
        total_chars=5_547_323,
        seed=2052,
    ),
    # Snapshot: xP3 English Generation (`xp3-en-generate`).
    # Dataset: xP3 English Generation (`bigscience/xP3`; provider: BigScience).
    # Summary: English question, context, passage, article, and long-form expansion instructions.
    # Purpose: Persist raw xP3 generation records for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `xp3-en-gen` raw datasource only.
    # Runtime task mapping: TASK_GENERATION in every task-facing view.
    # Budget: 371.952M train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "xp3-en-generate",
        source_id="xp3-en-gen",
        total_chars=381_952_280,
        seed=2052,
    ),
    # Snapshot: xP3 French Summarization (`xp3-fr-summary`).
    # Dataset: xP3 French Summarization (`bigscience/xP3`; provider: BigScience).
    # Summary: French article-condensation instructions rendered through thirteen xP3 templates.
    # Purpose: Persist raw xP3 summarization records for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `xp3-fr-sum` raw datasource only.
    # Runtime task mapping: TASK_SUMMARIZATION in every task-facing view.
    # Budget: 886.063M train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "xp3-fr-summary",
        source_id="xp3-fr-sum",
        total_chars=896_062_503,
        seed=2053,
    ),
    # Snapshot: XL-WiC French (`xlwic-fr`).
    # Dataset: XL-WiC French (`pasinit/xlwic`, xlwic_fr_fr; provider: XL-WiC authors).
    # Summary: French sentence pairs labelled by whether a target word has the same sense in both.
    # Purpose: Persist the selected raw pairs; natural/QCM variants are rendered in the training view.
    # Output format: Raw datasource records.
    # Composition: The `xlwic-fr` raw datasource only.
    # Runtime task mapping: TASK_CLASSIFICATION in every QCM training view.
    # Budget: 7.600M train chars; 379,983 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record once.
    unitary_raw_sft_snapshot(
        "xlwic-fr",
        source_id="xlwic-fr",
        total_chars=7_979_654,
        seed=2053,
    ),
    # Snapshot: xP3 French Generation (`xp3-fr-generate`).
    # Dataset: xP3 French Generation (`bigscience/xP3`; provider: BigScience).
    # Summary: French passage, article, long-form expansion, and headline generation instructions.
    # Purpose: Persist raw xP3 generation records for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `xp3-fr-gen` raw datasource only.
    # Runtime task mapping: TASK_GENERATION in every task-facing view.
    # Budget: 288.394M train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "xp3-fr-generate",
        source_id="xp3-fr-gen",
        total_chars=298_393_647,
        seed=2053,
    ),
    # Snapshot: XSum (`xsum`).
    # Dataset: XSum (`EdinburghNLP/xsum`, default; provider: EdinburghNLP).
    # Summary: BBC news articles paired with professionally written one-sentence summaries.
    # Purpose: Persist raw English article-summary pairs for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: English XSum articles and summaries.
    # Runtime task mapping: TASK_SUMMARIZATION in every task-facing view.
    # Budget: 465.339M train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Every indexed record exactly once; deterministic prompt wording varies by record id.
    unitary_raw_sft_snapshot(
        "xsum",
        source_id="xsum-en",
        total_chars=475_338_968,
        seed=2719,
    ),
    # Snapshot: OrangeSum Abstract (`orange-sum-abstract`).
    # Dataset: OrangeSum Abstract (`EdinburghNLP/orange_sum`, abstract; provider: EdinburghNLP).
    # Summary: French news articles paired with short professionally written abstracts.
    # Purpose: Persist raw French article-summary pairs for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: French OrangeSum Abstract articles and summaries.
    # Runtime task mapping: TASK_SUMMARIZATION in every task-facing view.
    # Budget: 49.272M train chars; 2.464M eval chars; together they cover all raw chars.
    # Selection: Every indexed record exactly once; deterministic prompt wording varies by record id.
    unitary_raw_sft_snapshot(
        "orange-sum-abstract",
        source_id="osum-fr-abs",
        total_chars=51_735_979,
        seed=2720,
    ),
    # Snapshot: OrangeSum Title (`orange-sum-title`).
    # Dataset: OrangeSum Title (`EdinburghNLP/orange_sum`, title; provider: EdinburghNLP).
    # Summary: French news articles paired with professionally written one-sentence titles.
    # Purpose: Persist raw French article-title pairs for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: French OrangeSum Title articles and headlines.
    # Runtime task mapping: TASK_GENERATION in every task-facing view.
    # Budget: 60.020M train chars; 3.001M eval chars; together they cover all expected raw chars.
    # Selection: Every indexed record exactly once; deterministic prompt wording varies by record id.
    unitary_raw_sft_snapshot(
        "orange-sum-title",
        source_id="osum-fr-title",
        total_chars=63_021_541,
        seed=2721,
    ),
    # Snapshot: xP3 English General Classification (`xp3-en-general-classification`).
    # Dataset: xP3 English General Classification (`bigscience/xP3`; provider: BigScience).
    # Summary: English WikiQA answer-validation and relevance classification templates.
    # Purpose: Persist raw xP3 classification records for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `xp3-en-gk-class` raw datasource only.
    # Runtime task mapping: TASK_CLASSIFICATION in every task-facing view.
    # Budget: 27.801M train chars; 1.390M eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "xp3-en-general-classification",
        source_id="xp3-en-gk-class",
        total_chars=29_191_238,
    ),
    # Snapshot: xP3 English General Generation (`xp3-en-general-generation`).
    # Dataset: xP3 English General Generation (`bigscience/xP3`; provider: BigScience).
    # Summary: English question, Jeopardy-style clue, and process-continuation generation templates.
    # Purpose: Persist raw xP3 generation records for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `xp3-en-gk-gen` raw datasource only.
    # Runtime task mapping: TASK_GENERATION in every task-facing view.
    # Budget: 21.915M train chars; 1.096M eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "xp3-en-general-generation",
        source_id="xp3-en-gk-gen",
        total_chars=23_010_258,
    ),
    # Snapshot: xP3 English Information Extraction (`xp3-en-information-extraction`).
    # Dataset: xP3 English Information Extraction (`bigscience/xP3`; provider: BigScience).
    # Summary: English WikiQA topic extraction from supplied questions and answers.
    # Purpose: Persist raw xP3 information-extraction records for explicit runtime SFT rendering.
    # Output format: Raw datasource records.
    # Composition: The `xp3-en-ie` raw datasource only.
    # Runtime task mapping: TASK_INFORMATION_EXTRACTION in every task-facing view.
    # Budget: 617,972 train chars; 30,898 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "xp3-en-information-extraction",
        source_id="xp3-en-ie",
        total_chars=648_870,
    ),
]
