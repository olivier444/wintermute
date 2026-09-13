from __future__ import annotations

from wintermute.data.get_raw.config import GetRawPreprocessorConfig

from ...model import DatasetPreset

ASSISTANT_SFT_DATASET_PRESETS = [
    # Dataset: UltraChat 200k (`HuggingFaceH4/ultrachat_200k`; provider: Hugging Face H4).
    # Summary: Filtered synthetic multi-turn instructional dialogues generated with ChatGPT.
    # Natural format: multi-turn conversational SFT.
    # Orientation: assistant, general instruction following.
    # Recommended SFT task spec: omitted; the source mixes tasks without task metadata.
    # Volume: large (207,865 train_sft conversations).
    DatasetPreset(
        uid="uch",
        name="ultrachat",
        dataset_name="HuggingFaceH4/ultrachat_200k",
        text_fields=("messages",),
        get_raw_output_max_shards=600,
        split="train_sft",
        lang=["en"],
        seed=2719,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),

    # Dataset: SmolTalk (`HuggingFaceTB/smol-smoltalk`; provider: Hugging Face TB).
    # Summary: Curated mixture of assistant conversations assembled from several instruction sources.
    # Natural format: multi-turn conversational SFT.
    # Orientation: assistant, instruction following, light reasoning.
    # Recommended SFT task spec: field `source`; smol-summarize-20k -> summarization;
    # smollm-rewrite-30k and explore-instruct-rewrite -> rewriting; self-oss-instruct
    # -> generation. Explicitly omit every other observed source, including the misleading
    # multi-turn smol-summarize-5k group.
    # Volume: large (roughly one million conversations).
    DatasetPreset(
        uid="smt",
        name="smoltalk",
        dataset_name="HuggingFaceTB/smol-smoltalk",
        text_fields=("messages",),
        get_raw_output_max_shards=600,
        lang=["en"],
        get_raw_stat_fields=["source"],
        seed=2219,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),
    # Dataset: Magpie-Pro-300K-Filtered (`Magpie-Align/Magpie-Pro-300K-Filtered`; provider: Magpie Align).
    # Summary: High-quality synthetic instruction-response conversations selected from Magpie-Pro.
    # Natural format: SFT (instruction to response).
    # Orientation: assistant, general instruction following.
    # !!CAUTION!! This dataset is oriented toward LONG answers
    # Recommended SFT task spec: omitted; the source has no reliable task metadata.
    # Volume: medium (300,000 conversations).
    DatasetPreset(
        uid="mpm",
        name="magpie-mono-pro-300k",
        dataset_name="Magpie-Align/Magpie-Pro-300K-Filtered",
        text_fields=("conversations",),
        get_raw_output_max_shards=600,
        lang=["EN"],
        seed=2219,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),
    # Dataset: Magpie Llama 3.1 Pro 300K (`Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered`; provider: Magpie Align).
    # Summary: Filtered Llama 3.1-generated instruction-response conversations with quality metadata.
    # Natural format: SFT (instruction to response).
    # Orientation: assistant, general instruction following.
    # !!CAUTION!! This dataset is oriented toward LONG answers
    # Recommended SFT task spec: field `task_category`, mapped to routing labels.
    # Mapping: Information seeking -> knowledge_qa; Math/Reasoning/Data analysis ->
    # problem_solving; Advice seeking, Role playing, Planning, Brainstorming, and Creative
    # writing -> generation. Editing and Coding & Debugging are explicitly omitted as mixed.
    # Use first-turn scope: task_category describes the initial instruction, and sampled
    # second turns sometimes change task type.
    # Volume: medium (300,000 conversations).
    DatasetPreset(
        uid="mlp3",
        name="magpie-llama-pro-300k",
        dataset_name="Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered",
        text_fields=("conversations",),
        get_raw_output_max_shards=600,
        lan_field="language",
        lang=["EN"],
        get_raw_stat_fields=["difficulty", "input_quality", "task_category"],
        seed=2219,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),
    # Dataset: SlimOrca (`Open-Orca/SlimOrca`; provider: Open-Orca).
    # Summary: Compact ShareGPT-style instruction conversations derived from OpenOrca data.
    # Natural format: multi-turn conversational SFT.
    # Orientation: assistant, QA, instruction following.
    # Recommended SFT task spec: omitted; the source mixes tasks without task metadata.
    # Volume: medium (about 518,000 conversations).
    DatasetPreset(
        uid="sor",
        name="slim-orca",
        dataset_name="Open-Orca/SlimOrca",
        text_fields=("conversations",),
        get_raw_output_max_shards=600,
        lang=["EN"],
        seed=2219,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),
    # Dataset: Everyday Conversations Llama 3.1 2K (`HuggingFaceTB/everyday-conversations-llama3.1-2k`; provider: Hugging Face TB).
    # Summary: Short, natural everyday dialogues for conversational assistant behaviour.
    # Natural format: multi-turn conversational SFT.
    # Orientation: assistant, casual conversation, brevity.
    # Recommended SFT task spec: omitted; topic metadata does not identify each turn's task intent.
    # Volume: small (about 2,000 conversations).
    DatasetPreset(
        uid="edc",
        name="everyday-conversations",
        dataset_name="HuggingFaceTB/everyday-conversations-llama3.1-2k",
        text_fields=("messages",),
        get_raw_output_max_shards=600,
        split="train_sft",
        lang=["EN"],
        get_raw_stat_fields=["topic", "subtopic"],
        seed=2219,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),
    # Dataset: No Robots (`HuggingFaceH4/no_robots`; provider: Hugging Face H4).
    # Summary: High-quality, human-written instruction-response examples without model generation.
    # Natural format: SFT (instruction to response).
    # Orientation: assistant, high-quality instruction following.
    # Recommended SFT task spec: field `category`; Summarize -> summarization;
    # Open QA -> knowledge_qa; Closed QA/Extract -> information_extraction; Classify ->
    # classification; Generation/Brainstorm/Coding -> generation; Rewrite -> rewriting;
    # Chat -> explicit omission.
    # Volume: small (about 10,000 examples).
    DatasetPreset(
        uid="nor",
        name="hf_no_robots",
        dataset_name="HuggingFaceH4/no_robots",
        text_fields=("messages",),
        get_raw_output_max_shards=600,
        lang=["en"],
        get_raw_stat_fields=["category"],
        seed=2219,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),
    # Dataset: Open-Platypus (`garage-bAInd/Open-Platypus`; provider: garage-bAInd).
    # Summary: Instruction mixture spanning STEM, logic, mathematics, and programming.
    # Natural format: SFT (instruction and optional context to response).
    # Orientation: assistant, reasoning, STEM, code.
    # Recommended SFT task spec: field `data_source`; MATH/PRM-800K, ARB, scibench,
    # theoremqa, and reclor -> problem_solving; scienceqa -> knowledge_qa; leetcode_ne
    # -> generation. Explicitly omit airoboros, guanaco, and tigerbot-kaggle.
    # Volume: small to medium (about 25,000 examples).
    DatasetPreset(
        uid="opy",
        name="Open-Platypus",
        dataset_name="garage-bAInd/Open-Platypus",
        text_fields=("instruction", "input", "output"),
        get_raw_output_max_shards=600,
        lang=["en"],
        get_raw_stat_fields=["data_source"],
        seed=2219,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),
    # Dataset: Databricks Dolly 15K (`databricks/databricks-dolly-15k`; provider: Databricks).
    # Summary: Employee-authored instructions, optional context, and responses.
    # Natural format: SFT (instruction and optional context to response).
    # Orientation: assistant, general instruction following.
    # Recommended SFT task spec: field `category`, mapped as follows: closed_qa and
    # information_extraction -> information_extraction; open_qa/general_qa -> knowledge_qa;
    # classification -> classification; summarization -> summarization;
    # brainstorming/creative_writing -> generation.
    # Volume: small (15,000 examples).
    DatasetPreset(
        uid="d15",
        name="dolly-15k",
        dataset_name="databricks/databricks-dolly-15k",
        text_fields=("context", "instruction", "response"),
        get_raw_output_max_shards=600,
        lang=["en"],
        get_raw_stat_fields=["category"],
        seed=2219,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
    ),
    # Dataset: OpenAssistant OASST1 (`OpenAssistant/oasst1`, English subset; provider: OpenAssistant).
    # Summary: Crowdsourced assistant conversation trees, flattened locally into training conversations.
    # Natural format: multi-turn conversational SFT.
    # Orientation: assistant alignment, English.
    # Recommended SFT task spec: omitted; task intent can change between turns and is not annotated.
    # Volume: medium (about 89,000 source messages; English subset used here).
    DatasetPreset(
        uid="oa1",
        name="oasst1",
        dataset_name="OpenAssistant/oasst1",
        text_fields=("messages",),
        get_raw_output_max_shards=600,
        lan_field="lang",
        lang=["en"],
        seed=2219,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
        get_raw_preprocessors=[GetRawPreprocessorConfig(kind="tree", params={})],
    ),
    # Dataset: OpenAssistant OASST2 (`OpenAssistant/oasst2`, French subset; provider: OpenAssistant).
    # Summary: Crowdsourced assistant conversation trees, flattened locally into training conversations.
    # Natural format: multi-turn conversational SFT.
    # Orientation: assistant alignment, French.
    # Recommended SFT task spec: omitted; task intent can change between turns and is not annotated.
    # Volume: small French subset (about 3,900 messages; 135,000 messages overall).
    DatasetPreset(
        uid="oa2-fr",
        name="oasst2-fr",
        dataset_name="OpenAssistant/oasst2",
        text_fields=("messages",),
        get_raw_output_max_shards=600,
        lan_field="lang",
        lang=["fr"],
        seed=2219,
        get_raw_shard_sample_count=2048,
        get_raw_shard_group_size=8,
        get_raw_preprocessors=[GetRawPreprocessorConfig(kind="tree", params={})],
    ),
]
