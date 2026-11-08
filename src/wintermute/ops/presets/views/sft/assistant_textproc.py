from __future__ import annotations

from ..builders import _composite_view_config, _view_component, _view_preset, _generic_view_component
from ..transforms import assistant, benchmarks, multiple_choice, reasoning, text_processing, translation

TXT_PROC = _composite_view_config(
    [
        # Translation: 10%.
        _view_component(
            "agentlans-en-fr-hq",
            0.070,
            max_restarts=0,
            record_transform=translation.agentlans_en_fr_hq_transform(),
        ),
        _view_component(
            "news-commentary-en-fr",
            0.030,
            max_restarts=0,
            record_transform=translation.news_commentary_en_fr_transform(),
        ),

        # Summarization: 10%.
        _view_component(
            "xp3-en-summary",
            0.030,
            max_restarts=0,
            record_transform=text_processing.xp3_en_summary_transform(),
        ),
        _view_component(
            "xp3-fr-summary",
            0.020,
            max_restarts=0,
            record_transform=text_processing.xp3_fr_summary_transform(),
        ),
        _view_component(
            "xsum",
            0.035,
            max_restarts=1,
            record_transform=text_processing.xsum_transform(),
        ),
        _view_component(
            "orange-sum-abstract",
            0.015,
            max_restarts=5,
            record_transform=text_processing.orange_sum_abstract_transform(),
        ),

        # Grounded QA / extraction: 18%.
        _view_component(
            "ufw-l3-qa",
            0.080,
            max_restarts=0,
            record_transform=text_processing.ultra_fineweb_l3_qa_transform(),
        ),
        _view_component(
            "xp3-en-qa",
            0.065,
            max_restarts=10,
            record_transform=text_processing.xp3_en_qa_transform(),
        ),
        _view_component(
            "boolq",
            0.010,
            max_restarts=10,
            record_transform=multiple_choice.boolq_transform(),
        ),
        _view_component(
            "babi_qa",
            0.004,
            max_restarts=10,
            record_transform=benchmarks.babi_qa_transform(),
        ),

        # Rewriting / semantic discrimination / short formats: 7%.
        _view_component(
            "xp3-en-paws-rewrite",
            0.025,
            max_restarts=10,
            record_transform=text_processing.xp3_en_paws_rewrite_transform(),
        ),
        _view_component(
            "orange-sum-title",
            0.010,
            max_restarts=6,
            record_transform=text_processing.orange_sum_title_transform(),
        ),
        _view_component(
            "ag_news",
            0.010,
            max_restarts=1,
            record_transform=multiple_choice.ag_news_transform(),
        ),
        _view_component(
            "paws",
            0.010,
            max_restarts=3,
            record_transform=multiple_choice.paws_transform(),
        ),
        _view_component(
            "wic",
            0.005,
            max_restarts=10,
            record_transform=multiple_choice.wic_transform(),
        ),
        _view_component(
            "xlwic-fr",
            0.010,
            max_restarts=2,
            record_transform=multiple_choice.xlwic_fr_transform(),
        ),
    ],
    seed=2053,
)

# 100M tokens
ASSISTANT = _composite_view_config(
    [
        _view_component(
            "hf_no_robots",
            0.04,
            max_restarts=5,
            record_transform=assistant.no_robots_transform(),
        ),
        _view_component(
            "smoltalk",
            0.3,
            max_restarts=0,
            record_transform=assistant.smoltalk_transform(),
        ),
        _view_component(
            "oasst1",
            0.07,
            max_restarts=2,
            record_transform=assistant.oasst1_transform(),
        ),
        _view_component(
            "oasst2-fr",
            0.019,
            max_restarts=10,
            record_transform=assistant.oasst2_fr_transform(),
        ),
        _view_component(
            "aya-fr",
            0.007,
            max_restarts=10,
            record_transform=assistant.aya_fr_transform(),
        ),
    ],
    seed=2053,
)

# 100M tokens
REASONING = _composite_view_config(
    [
        _view_component(
            "xp3-en-problem-solving",
            0.09,
            max_restarts=10,
            record_transform=reasoning.xp3_en_problem_solving_transform(),
        ),
        _view_component(
            "asdiv-a_svamp-raw",
            0.10,
            max_restarts=10,
            record_transform=benchmarks.asdiv_a_svamp_transform(),
        ),
        _view_component(
            "gsm8k",
            0.021,
            max_restarts=10,
            record_transform=benchmarks.gsm8k_transform(),
        ),
        _view_component(
            "MAWPS-raw",
            0.04,
            max_restarts=5,
            record_transform=benchmarks.mawps_transform(),
        ),
        _view_component(
            "winogrande",
            0.006,
            max_restarts=10,
            record_transform=multiple_choice.winogrande_transform(),
        ),
        _view_component(
            "piqa",
            0.02,
            max_restarts=10,
            record_transform=multiple_choice.piqa_transform(),
        ),
        _view_component(
            "commonsense_qa",
            0.006,
            max_restarts=10,
            record_transform=multiple_choice.commonsense_qa_transform(),
        ),
        _view_component(
            "ai2_arc",
            0.0015,
            max_restarts=10,
            record_transform=multiple_choice.ai2_arc_transform(),
        ),
        _view_component(
            "boolq",
            0.008,
            max_restarts=10,
            record_transform=multiple_choice.boolq_transform(),
        ),
        _view_component(
            "babi_qa",
            0.003,
            max_restarts=10,
            record_transform=benchmarks.babi_qa_transform(),
        ),
    ],
    seed=2053,
)


PRESET = _view_preset(
    "assistant-textproc",
    _composite_view_config(
        [
            _view_component(
                "mix40b-txt",
                0.25,
                include_prompt_in_loss=True,
                shard_offset_ratio=0.75,
            ),

            _generic_view_component(
                TXT_PROC,
                0.25,
                max_restarts=2
            ),

            _generic_view_component(
                ASSISTANT,
                0.25,
                max_restarts=2
            ),

            _generic_view_component(
                REASONING,
                0.25,
                max_restarts=2
            )
        ],
        seed=2053,
    )
)

