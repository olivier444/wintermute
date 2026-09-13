from __future__ import annotations

from ..builders import xp3_dataset
from ...model import DatasetPreset
from .xp3_sources import (
    XP3_EN_REWRITE_URLS,
    XP3_EN_TEXT_PROCESSING_URLS_BY_TASK,
    XP3_FR_TEXT_PROCESSING_URLS_BY_TASK,
)

TEXT_PROCESSING_SFT_DATASET_PRESETS = [
    # Dataset: xP3 English Summarization (`bigscience/xP3`; provider: BigScience).
    # Summary: English article-condensation instructions rendered through five xP3 templates.
    # Natural format: SFT (prompt to target).
    # Orientation: text transformation, English.
    # Recommended SFT task spec: `summarization` (constant).
    # Volume: five selected English xP3 task templates.
    xp3_dataset(
        uid="xp3-en-sum",
        name="xp3-en-summary",
        language="en",
        data_files=XP3_EN_TEXT_PROCESSING_URLS_BY_TASK["summary"],
        output_max_shards=200,
    ),
    # Dataset: xP3 English Question Answering (`bigscience/xP3`; provider: BigScience).
    # Summary: English extractive and passage-grounded question answering.
    # Natural format: SFT (prompt to target).
    # Orientation: reading comprehension, English.
    # Recommended SFT task spec: `information_extraction` (constant).
    # Volume: 22 selected English xP3 task templates.
    xp3_dataset(
        uid="xp3-en-qa",
        name="xp3-en-qa",
        language="en",
        data_files=XP3_EN_TEXT_PROCESSING_URLS_BY_TASK["qa"],
        output_max_shards=100,
    ),
    # Dataset: AG News (`fancyzhx/ag_news`, default; provider: Xiang Zhang).
    # Summary: English news articles classified into World, Sports, Business, or Science and Technology.
    # Natural format: structured four-class text classification.
    # Orientation: text classification, English.
    # Recommended SFT task spec: `classification` (constant).
    # Volume: medium (120,000 training examples).
    DatasetPreset(
        uid="ag-news",
        name="ag_news",
        dataset_name="fancyzhx/ag_news",
        text_fields=("text", "label"),
        get_raw_output_max_shards=100,
        config_template="default",
        lang=["en"],
        get_raw_stat_fields=["label"],
        seed=2052,
    ),
    # Dataset: PAWS (`google-research-datasets/paws`, labeled_final; provider: Google Research).
    # Summary: English sentence pairs labelled by whether they are paraphrases.
    # Natural format: structured binary text classification.
    # Orientation: text classification, English.
    # Recommended SFT task spec: `classification` (constant).
    # Volume: medium (49,401 training examples).
    DatasetPreset(
        uid="paws",
        name="paws",
        dataset_name="google-research-datasets/paws",
        text_fields=("sentence1", "sentence2", "label"),
        get_raw_output_max_shards=100,
        config_template="labeled_final",
        lang=["en"],
        get_raw_stat_fields=["label"],
        get_raw_exclude_fields=["id"],
        seed=2052,
    ),
    # Dataset: SuperGLUE WiC (`aps/super_glue`, wic; provider: SuperGLUE).
    # Summary: English sentence pairs labelled by whether a target word has the same sense in both.
    # Natural format: structured binary word-sense classification.
    # Orientation: text classification, English.
    # Recommended SFT task spec: `classification` (constant).
    # Volume: small (5,428 training examples).
    DatasetPreset(
        uid="wic",
        name="wic",
        dataset_name="aps/super_glue",
        text_fields=("word", "sentence1", "sentence2", "label"),
        get_raw_output_max_shards=100,
        config_template="wic",
        lang=["en"],
        get_raw_stat_fields=["label"],
        get_raw_exclude_fields=["idx", "start1", "start2", "end1", "end2"],
        seed=2052,
    ),
    # Dataset: xP3 English PAWS Rewriting (`bigscience/xP3`; provider: BigScience).
    # Summary: English sentence rewriting from the PAWS paraphrase-generation template.
    # Natural format: SFT (prompt to target).
    # Orientation: text rewriting, English.
    # Recommended SFT task spec: `rewriting` (constant).
    # Volume: one selected English xP3 task template.
    xp3_dataset(
        uid="xp3-en-paws-rewrite",
        name="xp3-en-paws-rewrite",
        language="en",
        data_files=XP3_EN_REWRITE_URLS,
        output_max_shards=100,
    ),
    # Dataset: xP3 English Generation (`bigscience/xP3`; provider: BigScience).
    # Summary: English question, context, passage, article, and long-form expansion instructions.
    # Natural format: SFT (prompt to target).
    # Orientation: open-ended text generation, English.
    # Recommended SFT task spec: `generation` (constant).
    # Volume: 13 selected English xP3 task templates.
    xp3_dataset(
        uid="xp3-en-gen",
        name="xp3-en-generate",
        language="en",
        data_files=XP3_EN_TEXT_PROCESSING_URLS_BY_TASK["generate"],
        output_max_shards=100,
    ),
    # Dataset: xP3 French Summarization (`bigscience/xP3`; provider: BigScience).
    # Summary: French article-condensation instructions rendered through thirteen xP3 templates.
    # Natural format: SFT (prompt to target).
    # Orientation: text transformation, French.
    # Recommended SFT task spec: `summarization` (constant).
    # Volume: 13 selected French xP3 task templates.
    xp3_dataset(
        uid="xp3-fr-sum",
        name="xp3-fr-summary",
        language="fr",
        data_files=XP3_FR_TEXT_PROCESSING_URLS_BY_TASK["summary"],
        output_max_shards=200,
    ),
    # Dataset: XL-WiC French (`pasinit/xlwic`, xlwic_fr_fr; provider: XL-WiC authors).
    # Summary: French sentence pairs labelled by whether a target word has the same sense in both.
    # Natural format: structured binary word-sense classification.
    # Orientation: text classification, French.
    # Recommended SFT task spec: `classification` (constant).
    # Volume: medium (French XL-WiC training split).
    DatasetPreset(
        uid="xlwic-fr",
        name="xlwic-fr",
        dataset_name="pasinit/xlwic",
        text_fields=("target_word", "context_1", "context_2", "label"),
        get_raw_output_max_shards=100,
        config_template="xlwic_fr_fr",
        lan_field="language",
        lang=["FR"],
        get_raw_stat_fields=["label"],
        get_raw_exclude_fields=[
            "id",
            "pos",
            "target_word_location_1",
            "target_word_location_2",
        ],
        seed=2053,
    ),
    # Dataset: xP3 French Generation (`bigscience/xP3`; provider: BigScience).
    # Summary: French passage, article, long-form expansion, and headline generation instructions.
    # Natural format: SFT (prompt to target).
    # Orientation: open-ended text generation, French.
    # Recommended SFT task spec: `generation` (constant).
    # Volume: nine selected French xP3 task templates.
    xp3_dataset(
        uid="xp3-fr-gen",
        name="xp3-fr-generate",
        language="fr",
        data_files=XP3_FR_TEXT_PROCESSING_URLS_BY_TASK["generate"],
        output_max_shards=100,
    ),
    # Dataset: XSum (`EdinburghNLP/xsum`, default; provider: EdinburghNLP).
    # Summary: BBC news articles paired with professionally written one-sentence summaries.
    # Natural format: SFT (document to abstractive summary).
    # Orientation: English news summarization.
    # Recommended SFT task spec: `summarization` (constant).
    # Volume: medium (204,045 training examples).
    DatasetPreset(
        uid="xsum-en",
        name="xsum",
        dataset_name="EdinburghNLP/xsum",
        text_fields=("document", "summary"),
        get_raw_output_max_shards=100,
        config_template="default",
        lang=["en"],
        get_raw_exclude_fields=["id"],
        seed=2719,
    ),
    # Dataset: OrangeSum Abstract (`EdinburghNLP/orange_sum`, abstract; provider: EdinburghNLP).
    # Summary: French news articles paired with short professionally written abstracts.
    # Natural format: SFT (document to abstractive summary).
    # Orientation: French news summarization.
    # Recommended SFT task spec: `summarization` (constant).
    # Volume: small (21,401 training examples).
    DatasetPreset(
        uid="osum-fr-abs",
        name="orange-sum-abstract",
        dataset_name="EdinburghNLP/orange_sum",
        text_fields=("text", "summary"),
        get_raw_output_max_shards=100,
        config_template="abstract",
        lang=["fr"],
        seed=2719,
    ),
    # Dataset: OrangeSum Title (`EdinburghNLP/orange_sum`, title; provider: EdinburghNLP).
    # Summary: French news articles paired with professionally written one-sentence titles.
    # Natural format: SFT (document to abstractive headline).
    # Orientation: French news headline generation.
    # Recommended SFT task spec: `generation` (constant).
    # Volume: small (30,659 training examples).
    DatasetPreset(
        uid="osum-fr-title",
        name="orange-sum-title",
        dataset_name="EdinburghNLP/orange_sum",
        text_fields=("text", "summary"),
        get_raw_output_max_shards=100,
        config_template="title",
        lang=["fr"],
        seed=2719,
    ),
]
