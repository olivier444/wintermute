from __future__ import annotations

from ..builders import cosmopedia_dataset
from ...model import DatasetPreset

SYNTHETIC_PRETRAIN_DATASET_PRESETS = [
    # Dataset: Cosmopedia Stories (`HuggingFaceTB/cosmopedia`; provider: Hugging Face TB).
    # Summary: Synthetic educational stories generated for language-model pretraining.
    # Natural format: classic NTP (text documents).
    # Orientation: foundation, synthetic educational narrative.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: large synthetic subcorpus.
    cosmopedia_dataset(uid="cst", name="cosmopedia-stories", config_template="stories"),
    # Dataset: Cosmopedia Khan Academy (`HuggingFaceTB/cosmopedia`; provider: Hugging Face TB).
    # Summary: Synthetic textbook-style educational content inspired by Khan Academy material.
    # Natural format: classic NTP (text documents).
    # Orientation: foundation, education, STEM.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: medium synthetic subcorpus.
    cosmopedia_dataset(uid="ckh", name="cosmopedia-kahn", config_template="khanacademy"),
    # Dataset: Cosmopedia OpenStax (`HuggingFaceTB/cosmopedia`; provider: Hugging Face TB).
    # Summary: Synthetic textbook-style educational content derived from OpenStax seed material.
    # Natural format: classic NTP (text documents).
    # Orientation: foundation, education, STEM.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: medium synthetic subcorpus.
    cosmopedia_dataset(uid="cox", name="cosmopedia-openstax", config_template="openstax"),
    # Dataset: Cosmopedia Auto Math Text (`HuggingFaceTB/cosmopedia`; provider: Hugging Face TB).
    # Summary: Synthetic mathematical exposition generated for pretraining.
    # Natural format: classic NTP (text documents).
    # Orientation: foundation, mathematics, STEM.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: large synthetic subcorpus.
    cosmopedia_dataset(uid="cmt", name="cosmopedia-auto_math_text", config_template="auto_math_text"),
    # Dataset: Cosmopedia WikiHow (`HuggingFaceTB/cosmopedia`; provider: Hugging Face TB).
    # Summary: Synthetic instructional articles based on WikiHow-style procedural seed material.
    # Natural format: classic NTP (text documents).
    # Orientation: foundation, procedural knowledge, instruction following.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: large synthetic subcorpus.
    cosmopedia_dataset(uid="cwh", name="cosmopedia-wikihow", config_template="wikihow"),
    # Dataset: TinyStories (`roneneldan/TinyStories`; provider: Ronen Eldan).
    # Summary: Synthetic short stories with simple language for studying small language models.
    # Natural format: classic NTP (stories).
    # Orientation: sanity-check corpus, simple language, narrative.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: medium (about 2 million stories).
    DatasetPreset(
        uid="tns",
        name="tinystories",
        dataset_name="roneneldan/TinyStories",
        text_fields=("text",),
        get_raw_output_max_shards=300,
        config_template="default",
    ),
]
