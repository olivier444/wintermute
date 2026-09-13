from __future__ import annotations

from collections.abc import Iterator

from wintermute.data.record import DataRecord
from wintermute.data.snapshot.materialized_iterable import AbstractMaterializedView
from wintermute.data.transform import RecordTransform
from wintermute.data.transform.context import (
    DEFAULT_RUNTIME_TRANSFORM_SEED,
    TransformContext,
    new_runtime_transform_rng,
)


class TransformedMaterializedView(AbstractMaterializedView):
    def __init__(
        self,
        wrapped: AbstractMaterializedView,
        transform: RecordTransform,
        *,
        seed: int = DEFAULT_RUNTIME_TRANSFORM_SEED,
    ) -> None:
        self.wrapped = wrapped
        self.transform = transform
        self._rng = new_runtime_transform_rng(seed)
        self.iteration = 0

    @property
    def name(self) -> str:
        return self.wrapped.name

    def __iter__(self) -> Iterator[DataRecord]:
        iteration = self.iteration
        self.iteration += 1
        context = TransformContext(rng=self._rng, iteration=iteration)
        for record in self.wrapped:
            yield from self.transform.transform(record, context=context)
