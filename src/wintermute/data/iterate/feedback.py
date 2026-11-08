from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Iterator, Literal, Protocol, TypeVar

from wintermute.data.constants import (
    FLD_GENERIC_COMPLETION,
    FLD_GENERIC_PROMPT,
    FLD_GENERIC_SYSTEM_PROMPT,
)
from wintermute.data.iterate.transformed_materialized_view import TransformedMaterializedView
from wintermute.data.snapshot.materialized_iterable import AbstractMaterializedView
from wintermute.data.transform import RecordTransform, RecordTransformConfig, SequentialTransform
from wintermute.data.transform.builder import build_record_transform
from wintermute.data.transform.context import DEFAULT_RUNTIME_TRANSFORM_SEED
from wintermute.data.transform.implementations.normalize_single_turn_sft import NormalizeSingleTurnSftRecord
from wintermute.ml.tokenization.chat_format import ConversationContext, ConversationTurn

ScoreKind = Literal["absolute", "relative"]


@dataclass(frozen=True)
class FeedbackPrompt:
    uid: str
    context: ConversationContext
    reference_completion: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MaterializedFeedbackPromptSource:
    """Project materialized SFT records into prompts for online feedback."""

    def __init__(
        self,
        materialized_view: AbstractMaterializedView,
        *,
        record_transform: RecordTransformConfig | None = None,
        transform_seed: int = DEFAULT_RUNTIME_TRANSFORM_SEED,
    ) -> None:
        transform: RecordTransform = NormalizeSingleTurnSftRecord()
        if record_transform is not None:
            transform = SequentialTransform(
                [build_record_transform(record_transform), transform]
            )
        self.materialized_view = TransformedMaterializedView(
            materialized_view,
            transform,
            seed=transform_seed,
        )

    def __iter__(self) -> Iterator[FeedbackPrompt]:
        for record in self.materialized_view:
            fields = record.fields
            if f"{FLD_GENERIC_PROMPT}.0" not in fields:
                raise ValueError(
                    f"materialized record '{record.record_id}' has no SFT prompt"
                )

            system = fields.get(FLD_GENERIC_SYSTEM_PROMPT)
            if system is not None and not isinstance(system, str):
                raise TypeError(
                    f"materialized record '{record.record_id}' has a non-text system prompt"
                )

            history: list[ConversationTurn] = []
            index = 0
            while (prompt_key := f"{FLD_GENERIC_PROMPT}.{index}") in fields:
                prompt = fields[prompt_key]
                completion = fields.get(f"{FLD_GENERIC_COMPLETION}.{index}")
                if not isinstance(prompt, str):
                    raise TypeError(
                        f"materialized record '{record.record_id}' has a non-text prompt at turn {index}"
                    )
                if completion is not None and not isinstance(completion, str):
                    raise TypeError(
                        f"materialized record '{record.record_id}' has a non-text completion at turn {index}"
                    )
                if completion is None and f"{FLD_GENERIC_PROMPT}.{index + 1}" in fields:
                    raise ValueError(
                        f"materialized record '{record.record_id}' is missing the completion needed for turn {index + 1}"
                    )

                yield FeedbackPrompt(
                    uid=f"{record.record_id}.{index}",
                    context=ConversationContext(
                        current_user=prompt,
                        history=tuple(history),
                        system=system,
                    ),
                    reference_completion=completion,
                )

                if completion is None:
                    break
                history.append(ConversationTurn(user=prompt, assistant=completion))
                index += 1


@dataclass(frozen=True)
class GeneratedCompletion:
    completion: str
    trace: object | None = None


@dataclass(frozen=True)
class ScoredCompletion:
    completion: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: object | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("feedback score must be finite")


@dataclass(frozen=True)
class FeedbackExample:
    prompt: FeedbackPrompt
    completions: tuple[ScoredCompletion, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    score_kind: ScoreKind = "absolute"

    def __post_init__(self) -> None:
        if len(self.completions) < 2:
            raise ValueError("feedback example requires at least two completions")
        if self.score_kind not in ("absolute", "relative"):
            raise ValueError("feedback score kind must be absolute or relative")


@dataclass(frozen=True)
class FeedbackTrainingBatch:
    examples: tuple[FeedbackExample, ...]

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("feedback training batch cannot be empty")


SourceItemT = TypeVar("SourceItemT")
ProducerItemT = TypeVar("ProducerItemT", contravariant=True)


class FeedbackBatchProducer(Protocol[ProducerItemT]):
    """Turns source items into scored batches using its own batching policy."""

    def iter_batches(
        self,
        source: Iterable[ProducerItemT],
    ) -> Iterator[FeedbackTrainingBatch]: ...


class AbstractFeedbackTrainingView(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[FeedbackTrainingBatch]: ...


class FeedbackTrainingView(AbstractFeedbackTrainingView, Generic[SourceItemT]):
    def __init__(
        self,
        source: Iterable[SourceItemT],
        batch_producer: FeedbackBatchProducer[SourceItemT],
    ) -> None:
        self.source = source
        self.batch_producer = batch_producer

    def __iter__(self) -> Iterator[FeedbackTrainingBatch]:
        yield from self.batch_producer.iter_batches(self.source)
