from __future__ import annotations

from .builders import _view_preset, _snapshot_view_config
from .transforms import general_knowledge, multiple_choice, reasoning, text_processing, translation


PRESETS = [
    _view_preset(
        "default4-eval-pretrain",
        _snapshot_view_config("default4-txt", split="eval3", include_prompt_in_loss=True),
    ),
    _view_preset("default4-eval", _snapshot_view_config("default4-txt", split="eval3")),
    _view_preset("abstraction-id-eval", _snapshot_view_config("abstraction-v1", split="valid_id")),
    _view_preset("abstraction-ood-eval", _snapshot_view_config("abstraction-v1", split="valid_ood")),
    _view_preset("default1-sft-eval", _snapshot_view_config("default1-sft", split="eval")),
    _view_preset("default2-sft-eval", _snapshot_view_config("default2-sft", split="eval")),
    _view_preset(
        "default5-eval-pretrain",
        _snapshot_view_config("default5-txt", split="eval", include_prompt_in_loss=True),
    ),
    _view_preset("default5-eval", _snapshot_view_config("default5-txt", split="eval")),
    _view_preset(
        "default6-eval-pretrain-deprecated",
        _snapshot_view_config(
            "default6-txt.deprecated", split="eval", include_prompt_in_loss=True
        ),
    ),
    _view_preset(
        "abstraction-sft-eval", _snapshot_view_config("abstraction1-sft", split="eval")
    ),
    _view_preset(
        "mix40b-eval-pretrain",
        _snapshot_view_config("mix40b-txt", split="eval", include_prompt_in_loss=True),
    ),
    _view_preset("mix40b-eval", _snapshot_view_config("mix40b-txt", split="eval")),
    _view_preset(
        "dclm-eval-pretrain",
        _snapshot_view_config("dclm-eval-txt", split="eval", include_prompt_in_loss=True),
    ),
    _view_preset(
        "code-eval-pretrain",
        _snapshot_view_config("code-txt", split="eval", include_prompt_in_loss=True),
    ),
    _view_preset("code-eval", _snapshot_view_config("code-txt", split="eval")),
    _view_preset(
        "french-eval-pretrain",
        _snapshot_view_config("french-txt", split="eval", include_prompt_in_loss=True),
    ),
    _view_preset("french-eval", _snapshot_view_config("french-txt", split="eval")),
    _view_preset(
        "news-commentary-eval",
        _snapshot_view_config(
            "news-commentary-en-fr",
            split="eval",
            record_transform=translation.news_commentary_en_fr_transform(),
        ),
    ),
    _view_preset(
        "xsum-eval",
        _snapshot_view_config(
            "xsum", split="eval", record_transform=text_processing.xsum_transform()
        ),
    ),
    _view_preset(
        "problem-solving-eval",
        _snapshot_view_config(
            "xp3-en-problem-solving",
            split="eval",
            record_transform=reasoning.xp3_en_problem_solving_transform(),
        ),
    ),
    _view_preset(
        "boolq-eval",
        _snapshot_view_config(
            "boolq", split="eval", record_transform=multiple_choice.boolq_transform()
        ),
    ),
    _view_preset(
        "ag-news-eval",
        _snapshot_view_config(
            "ag_news", split="eval", record_transform=multiple_choice.ag_news_transform()
        ),
    ),
    _view_preset(
        "generation-eval",
        _snapshot_view_config(
            "xp3-en-generate",
            split="eval",
            record_transform=text_processing.xp3_en_generate_transform(),
        ),
    ),
    _view_preset(
        "knowledge-qa-eval",
        _snapshot_view_config(
            "xp3-en-knowledge-qa",
            split="eval",
            record_transform=general_knowledge.xp3_en_knowledge_qa_transform(),
        ),
    ),
    _view_preset(
        "rewriting-eval",
        _snapshot_view_config(
            "xp3-en-paws-rewrite",
            split="eval",
            record_transform=text_processing.xp3_en_paws_rewrite_transform(),
        ),
    ),
    _view_preset(
        "agentlans-eval",
        _snapshot_view_config(
            "agentlans-en-fr-hq",
            split="eval",
            record_transform=translation.agentlans_en_fr_hq_transform(),
        ),
    ),
    _view_preset(
        "orange-summary-eval",
        _snapshot_view_config(
            "orange-sum-abstract",
            split="eval",
            record_transform=text_processing.orange_sum_abstract_transform(),
        ),
    ),
]
