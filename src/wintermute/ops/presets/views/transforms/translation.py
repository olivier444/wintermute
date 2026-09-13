from __future__ import annotations

from wintermute.data.constants import TASK_TRANSLATION
from wintermute.data.transform import RecordTransformConfig
from wintermute.data.transform.config_builders import fan_out, to_sft

from .prompts.translation import (
    _EN_TO_FR_TRANSLATION_INSTRUCTIONS,
    _FR_TO_EN_TRANSLATION_INSTRUCTIONS,
)


def agentlans_en_fr_hq_transform() -> RecordTransformConfig:
    return fan_out(
        to_sft(
            prompt=[
                {"random_text": _EN_TO_FR_TRANSLATION_INSTRUCTIONS},
                {"field": "english"},
            ],
            completion=[{"field": "french"}],
            task=TASK_TRANSLATION,
        ),
        to_sft(
            prompt=[
                {"random_text": _FR_TO_EN_TRANSLATION_INSTRUCTIONS},
                {"field": "french"},
            ],
            completion=[{"field": "english"}],
            task=TASK_TRANSLATION,
        ),
    )


def news_commentary_en_fr_transform() -> RecordTransformConfig:
    return fan_out(
        to_sft(
            prompt=[
                {"random_text": _EN_TO_FR_TRANSLATION_INSTRUCTIONS},
                {"field": "translation.en"},
            ],
            completion=[{"field": "translation.fr"}],
            task=TASK_TRANSLATION,
        ),
        to_sft(
            prompt=[
                {"random_text": _FR_TO_EN_TRANSLATION_INSTRUCTIONS},
                {"field": "translation.fr"},
            ],
            completion=[{"field": "translation.en"}],
            task=TASK_TRANSLATION,
        ),
    )
