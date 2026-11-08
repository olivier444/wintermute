from __future__ import annotations

from ..builders import _composite_view_config, _view_component, _view_preset
from ..transforms import benchmarks, multiple_choice, text_processing, translation


PRESET = _view_preset(
    "text-processing-sft",
    _composite_view_config(
        [
            # Pretraining anchor: same source mixture as the base pretrain.
            _view_component(
                "mix40b-txt",
                0.20,
                include_prompt_in_loss=True,
                shard_offset_ratio=0.5,
            ),
            # Translation: bidirectional EN/FR, grounded entirely in supplied text.
            _view_component(
                "agentlans-en-fr-hq",
                0.113,
                max_restarts=4,
                record_transform=translation.agentlans_en_fr_hq_transform(),
            ),
            _view_component(
                "news-commentary-en-fr",
                0.047,
                max_restarts=8,
                record_transform=translation.news_commentary_en_fr_transform(),
            ),
            # Summarization: kept at parity with translation (16% each).
            _view_component(
                "xp3-en-summary",
                0.049,
                max_restarts=4,
                record_transform=text_processing.xp3_en_summary_transform(),
            ),
            _view_component(
                "xp3-fr-summary",
                0.035,
                max_restarts=4,
                record_transform=text_processing.xp3_fr_summary_transform(),
            ),
            _view_component(
                "xsum",
                0.055,
                max_restarts=8,
                record_transform=text_processing.xsum_transform(),
            ),
            _view_component(
                "orange-sum-abstract",
                0.021,
                max_restarts=25,
                record_transform=text_processing.orange_sum_abstract_transform(),
            ),
            # Grounded QA / information extraction: the answer is supported by supplied context.
            _view_component(
                "ufw-l3-qa",
                0.144,
                max_restarts=1,
                record_transform=text_processing.ultra_fineweb_l3_qa_transform(),
            ),
            _view_component(
                "xp3-en-qa",
                0.128,
                max_restarts=8,
                record_transform=text_processing.xp3_en_qa_transform(),
            ),
            _view_component(
                "boolq",
                0.032,
                max_restarts=40,
                record_transform=multiple_choice.boolq_transform(),
            ),
            _view_component(
                "babi_qa",
                0.016,
                max_restarts=100,
                record_transform=benchmarks.babi_qa_transform(),
            ),
            # Same-language transformation and short grounded generation.
            _view_component(
                "xp3-en-paws-rewrite",
                0.064,
                max_restarts=100,
                record_transform=text_processing.xp3_en_paws_rewrite_transform(),
            ),
            _view_component(
                "orange-sum-title",
                0.024,
                max_restarts=25,
                record_transform=text_processing.orange_sum_title_transform(),
            ),
            # Small classification tail for semantic decisions and short output formats.
            _view_component(
                "ag_news",
                0.024,
                max_restarts=20,
                record_transform=multiple_choice.ag_news_transform(),
            ),
            _view_component(
                "paws",
                0.020,
                max_restarts=30,
                record_transform=multiple_choice.paws_transform(),
            ),
            _view_component(
                "wic",
                0.004,
                max_restarts=100,
                record_transform=multiple_choice.wic_transform(),
            ),
            _view_component(
                "xlwic-fr",
                0.024,
                max_restarts=50,
                record_transform=multiple_choice.xlwic_fr_transform(),
            ),
        ],
        seed=2053,
    ),
)
