from __future__ import annotations

from .common import unitary_raw_sft_snapshot


ASSISTANT_SFT_SNAPSHOT_PRESETS = [
    # Snapshot: UltraChat (`ultrachat`).
    # Dataset: UltraChat 200k (`HuggingFaceH4/ultrachat_200k`; provider: Hugging Face H4).
    # Summary: Filtered synthetic multi-turn instructional dialogues generated with ChatGPT.
    # Purpose: Persist the selected raw UltraChat records; SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `uch` raw datasource only.
    # Runtime task mapping: Omitted because the source mixes tasks without reliable task metadata.
    # Budget: 1.228B train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "ultrachat",
        source_id="uch",
        total_chars=1_238_119_990,
        seed=2219,
    ),
    # Snapshot: SmolTalk (`smoltalk`).
    # Dataset: SmolTalk (`HuggingFaceTB/smol-smoltalk`; provider: Hugging Face TB).
    # Summary: Curated mixture of assistant conversations assembled from several instruction sources.
    # Purpose: Persist the selected raw SmolTalk records; SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `smt` raw datasource only.
    # Runtime task mapping: Normalize reliable `source` values; explicitly omit reviewed ambiguous groups.
    # Budget: 1.852B train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "smoltalk",
        source_id="smt",
        total_chars=1_862_107_675,
        seed=2219,
    ),
    # Snapshot: Magpie Pro 300K (`magpie-mono-pro-300k`).
    # Dataset: Magpie-Pro-300K-Filtered (`Magpie-Align/Magpie-Pro-300K-Filtered`; provider: Magpie Align).
    # Summary: High-quality synthetic instruction-response conversations selected from Magpie-Pro.
    # Purpose: Persist the selected raw Magpie Pro records; SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `mpm` raw datasource only.
    # Runtime task mapping: Omitted because the source has no reliable task metadata.
    # Budget: 1.025B train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "magpie-mono-pro-300k",
        source_id="mpm",
        total_chars=1_034_977_722,
        seed=2219,
    ),
    # Snapshot: Magpie Llama 3.1 Pro 300K (`magpie-llama-pro-300k`).
    # Dataset: Magpie Llama 3.1 Pro 300K (`Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered`; provider: Magpie Align).
    # Summary: Filtered Llama 3.1-generated instruction-response conversations with quality metadata.
    # Purpose: Persist the selected raw Magpie Llama records; SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `mlp3` raw datasource only.
    # Runtime task mapping: Normalize record-level `task_category` on every assistant turn.
    # Budget: 1.455B train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "magpie-llama-pro-300k",
        source_id="mlp3",
        total_chars=1_464_945_332,
        seed=2219,
    ),
    # Snapshot: SlimOrca (`slim-orca`).
    # Dataset: SlimOrca (`Open-Orca/SlimOrca`; provider: Open-Orca).
    # Summary: Compact ShareGPT-style instruction conversations derived from OpenOrca data.
    # Purpose: Persist the selected raw SlimOrca records; SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `sor` raw datasource only.
    # Runtime task mapping: Omitted because the source mixes tasks without task metadata.
    # Budget: 956.354M train chars; 9,999,999 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "slim-orca",
        source_id="sor",
        total_chars=966_353_745,
        seed=2219,
    ),
    # Snapshot: Everyday Conversations (`everyday-conversations`).
    # Dataset: Everyday Conversations Llama 3.1 2K (`HuggingFaceTB/everyday-conversations-llama3.1-2k`; provider: Hugging Face TB).
    # Summary: Short, natural everyday dialogues for conversational assistant behaviour.
    # Purpose: Persist the selected raw everyday-conversation records; SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `edc` raw datasource only.
    # Runtime task mapping: Omitted because topic metadata does not identify each turn's task intent.
    # Budget: 2.087M train chars; 104,330 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "everyday-conversations",
        source_id="edc",
        total_chars=2_190_939,
        seed=2219,
    ),
    # Snapshot: No Robots (`hf_no_robots`).
    # Dataset: No Robots (`HuggingFaceH4/no_robots`; provider: Hugging Face H4).
    # Summary: High-quality, human-written instruction-response examples without model generation.
    # Purpose: Persist the selected raw No Robots records; SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `nor` raw datasource only.
    # Runtime task mapping: Normalize reliable `category` values; explicitly omit Chat.
    # Budget: 11.925M train chars; 596,260 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "hf_no_robots",
        source_id="nor",
        total_chars=12_521_461,
        seed=2219,
    ),
    # Snapshot: Open-Platypus (`Open-Platypus`).
    # Dataset: Open-Platypus (`garage-bAInd/Open-Platypus`; provider: garage-bAInd).
    # Summary: Instruction mixture spanning STEM, logic, mathematics, and programming.
    # Purpose: Persist the selected raw Open-Platypus records; SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `opy` raw datasource only.
    # Runtime task mapping: Normalize reliable `data_source` values; explicitly omit ambiguous groups.
    # Budget: 28.754M train chars; 1.438M eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "Open-Platypus",
        source_id="opy",
        total_chars=30_191_934,
        seed=2219,
    ),
    # Snapshot: Dolly 15K (`dolly-15k`).
    # Dataset: Databricks Dolly 15K (`databricks/databricks-dolly-15k`; provider: Databricks).
    # Summary: Employee-authored instructions, optional context, and responses.
    # Purpose: Persist the selected raw Dolly records; SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `d15` raw datasource only.
    # Runtime task mapping: Normalize every observed `category` value to a shared task label.
    # Budget: 11.249M train chars; 562,442 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "dolly-15k",
        source_id="d15",
        total_chars=11_811_291,
        seed=2219,
    ),
    # Snapshot: OpenAssistant OASST1 (`oasst1`).
    # Dataset: OpenAssistant OASST1 (`OpenAssistant/oasst1`, English subset; provider: OpenAssistant).
    # Summary: Crowdsourced assistant conversation trees, flattened locally into training conversations.
    # Purpose: Persist the selected raw English OASST1 records; SFT rendering happens in the training view.
    # Output format: Raw datasource records.
    # Composition: The `oa1` raw datasource only.
    # Runtime task mapping: Omitted because task intent can change between turns and is not annotated.
    # Budget: 32.589M train chars; 1.629M eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "oasst1",
        source_id="oa1",
        total_chars=34_217_938,
        seed=2219,
    ),
    # Snapshot: OpenAssistant OASST2 French (`oasst2-fr`).
    # Dataset: OpenAssistant OASST2 (`OpenAssistant/oasst2`, French subset; provider: OpenAssistant).
    # Summary: Crowdsourced assistant conversation trees, flattened locally into training conversations.
    # Purpose: Persist the selected raw French OASST2 records; SFT rendering belongs to an explicit future view.
    # Output format: Raw datasource records.
    # Composition: The `oa2-fr` raw datasource only.
    # Runtime task mapping: Omitted because task intent can change between turns and is not annotated.
    # Budget: 2.323M train chars; 116,157 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "oasst2-fr",
        source_id="oa2-fr",
        total_chars=2_439_316,
        seed=2219,
    ),
    # Snapshot: Aya French (`aya-fr`).
    # Dataset: Aya (`CohereLabs/aya_dataset`; provider: Cohere For AI).
    # Summary: Multilingual instruction-response pairs, filtered here to French.
    # Purpose: Persist the selected raw French Aya records; SFT rendering belongs to an explicit future view.
    # Output format: Raw datasource records.
    # Composition: The `aya-fr` raw datasource only.
    # Runtime task mapping: Omitted because `annotation_type` describes provenance, not task intent.
    # Budget: 784,712 train chars; 39,235 eval chars; together they cover all raw chars.
    # Selection: Every indexed source record is eligible once.
    unitary_raw_sft_snapshot(
        "aya-fr",
        source_id="aya-fr",
        total_chars=823_947,
    ),
]
