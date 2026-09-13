from __future__ import annotations

from wintermute.data.transform import RecordTransformConfig
from wintermute.data.transform.config_builders import lookup_retrieval

from wintermute.data.iterate.dataclasses import CompositeTrainingSource

from .builders import _composite_view_config, _view_preset, _view_component
from .transforms import (
    assistant,
    benchmarks,
    general_knowledge,
    multiple_choice,
    reasoning,
    text_processing,
    translation,
)


def _task_routing_component(
    snapshot_id: str,
    weight: float,
    *,
    record_transform: RecordTransformConfig | None = None,
) -> CompositeTrainingSource:
    return _view_component(
        snapshot_id,
        weight,
        max_restarts={
            "ag_news": 10,
            "paws": 10,
            "wic": 10,
            "xlwic-fr": 10,
        }.get(snapshot_id, 6),
        record_transform=record_transform,
    )


PRESETS = [
    _view_preset(
        "legacy-sft-1",
        _composite_view_config(
            [
                _view_component("default1-sft", 0.9),
                _view_component("default4-txt", 0.08),
                _view_component("abstraction-v1", 0.02),
            ],
            seed=7312,
        ),
    ),
    _view_preset(
        "legacy-sft-2",
        _composite_view_config(
            [
                _view_component("default2-sft", 0.85, max_restarts=1),
                _view_component("default4-txt", 0.13),
                _view_component("abstraction-v1", 0.02),
            ],
            seed=7312,
        ),
    ),
    _view_preset(
        "abstraction-sft",
        _composite_view_config(
            [
                _view_component("abstraction-v1", 0.7),
                _view_component("abstraction1-sft", 0.4, max_restarts=9),
                _view_component("default2-sft", 0.2, max_restarts=1),
                _view_component("default5-txt", 0.4),
            ],
            seed=7312,
        ),
    ),
    _view_preset(
        "assistant-sft",
        _composite_view_config(
            [
                _view_component("abstraction1-sft", 0.05, max_restarts=10),
                _view_component("default2-sft", 0.75, max_restarts=1),
                _view_component("default5-txt", 0.2),
            ],
            seed=7312,
        ),
    ),
    _view_preset(
        "assistant-sft-extended",
        _composite_view_config(
            [
                _view_component("abstraction1-sft", 0.1, max_restarts=20),
                _view_component("default2-sft", 0.7, max_restarts=1),
                _view_component("default5-txt", 0.2),
            ],
            seed=7312,
        ),
    ),
    _view_preset(
        "task-routing",
        _composite_view_config(
            [
                _view_component(
                    "mix40b-txt",
                    0.338,
                    include_prompt_in_loss=True,
                    shard_offset_ratio=0.5,
                ),
                _view_component(
                    "mix40b-txt",
                    0.006,
                    max_restarts=100,
                    shard_offset_ratio=0.5,
                    record_transform=lookup_retrieval(),
                ),
                _task_routing_component("smoltalk", 0.103, record_transform=assistant.smoltalk_transform()),
                _task_routing_component("ultrachat", 0.035, record_transform=assistant.ultrachat_transform()),
                _task_routing_component("oasst1", 0.033, record_transform=assistant.oasst1_transform()),
                _task_routing_component(
                    "Open-Platypus", 0.022, record_transform=assistant.open_platypus_transform()
                ),
                _task_routing_component(
                    "hf_no_robots", 0.011, record_transform=assistant.no_robots_transform()
                ),
                _task_routing_component("dolly-15k", 0.005, record_transform=assistant.dolly_15k_transform()),
                _task_routing_component(
                    "magpie-mono-pro-300k", 0.015, record_transform=assistant.magpie_mono_pro_transform()
                ),
                _task_routing_component(
                    "magpie-llama-pro-300k", 0.015, record_transform=assistant.magpie_llama_pro_transform()
                ),
                _task_routing_component("slim-orca", 0.0095, record_transform=assistant.slim_orca_transform()),
                _task_routing_component(
                    "everyday-conversations",
                    0.0015,
                    record_transform=assistant.everyday_conversations_transform(),
                ),
                _task_routing_component(
                    "agentlans-en-fr-hq",
                    0.07272727,
                    record_transform=translation.agentlans_en_fr_hq_transform(),
                ),
                _task_routing_component(
                    "news-commentary-en-fr",
                    0.02727273,
                    record_transform=translation.news_commentary_en_fr_transform(),
                ),
                _task_routing_component(
                    "xp3-en-summary", 0.03977519, record_transform=text_processing.xp3_en_summary_transform()
                ),
                _task_routing_component(
                    "xp3-fr-summary", 0.01809715, record_transform=text_processing.xp3_fr_summary_transform()
                ),
                _task_routing_component("xsum", 0.03319149, record_transform=text_processing.xsum_transform()),
                _task_routing_component(
                    "orange-sum-abstract",
                    0.00510639,
                    record_transform=text_processing.orange_sum_abstract_transform(),
                ),
                _task_routing_component(
                    "xp3-en-paws-rewrite",
                    0.00382979,
                    record_transform=text_processing.xp3_en_paws_rewrite_transform(),
                ),
                _task_routing_component(
                    "xp3-en-problem-solving",
                    0.02317073,
                    record_transform=reasoning.xp3_en_problem_solving_transform(),
                ),
                _task_routing_component("MAWPS-raw", 0.00243903, record_transform=benchmarks.mawps_transform()),
                _task_routing_component("gsm8k", 0.00609756, record_transform=benchmarks.gsm8k_transform()),
                _task_routing_component(
                    "asdiv-a_svamp-raw", 0.01768293, record_transform=benchmarks.asdiv_a_svamp_transform()
                ),
                _task_routing_component("winogrande", 0.002, record_transform=multiple_choice.winogrande_transform()),
                _task_routing_component(
                    "xp3-en-qa", 0.00935714, record_transform=text_processing.xp3_en_qa_transform()
                ),
                _task_routing_component("boolq", 0.001, record_transform=multiple_choice.boolq_transform()),
                _task_routing_component("babi_qa", 0.00014286, record_transform=benchmarks.babi_qa_transform()),
                _task_routing_component(
                    "xp3-en-information-extraction",
                    0.00014286,
                    record_transform=text_processing.xp3_en_information_extraction_transform(),
                ),
                _task_routing_component("ag_news", 0.02608695, record_transform=multiple_choice.ag_news_transform()),
                _task_routing_component(
                    "xp3-en-general-classification",
                    0.001,
                    record_transform=text_processing.xp3_en_general_classification_transform(),
                ),
                _task_routing_component("paws", 0.01652174, record_transform=multiple_choice.paws_transform()),
                _task_routing_component("wic", 0.00173913, record_transform=multiple_choice.wic_transform()),
                _task_routing_component("xlwic-fr", 0.01304348, record_transform=multiple_choice.xlwic_fr_transform()),
                _task_routing_component(
                    "xp3-en-generate", 0.03635037, record_transform=text_processing.xp3_en_generate_transform()
                ),
                _task_routing_component(
                    "xp3-en-general-generation",
                    0.00262774,
                    record_transform=text_processing.xp3_en_general_generation_transform(),
                ),
                _task_routing_component(
                    "xp3-fr-generate", 0.01751826, record_transform=text_processing.xp3_fr_generate_transform()
                ),
                _task_routing_component(
                    "orange-sum-title",
                    0.00350363,
                    record_transform=text_processing.orange_sum_title_transform(),
                ),
                _task_routing_component(
                    "xp3-en-knowledge-qa",
                    0.03646666,
                    record_transform=general_knowledge.xp3_en_knowledge_qa_transform(),
                ),
                _task_routing_component(
                    "commonsense_qa", 0.002, record_transform=multiple_choice.commonsense_qa_transform()
                ),
                _task_routing_component("ai2_arc", 0.0006, record_transform=multiple_choice.ai2_arc_transform()),
                _task_routing_component("piqa", 0.007, record_transform=multiple_choice.piqa_transform()),
            ],
            seed=2053,
        ),
    ),
]
