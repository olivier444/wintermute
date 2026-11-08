from __future__ import annotations

from collections.abc import Iterable, Iterator

from wintermute.data.iterate.dataclasses import TrainingExample


def limit_eval_examples(
    examples: Iterable[TrainingExample],
    max_eval_units: int | None,
) -> Iterator[TrainingExample]:
    if max_eval_units is None:
        yield from examples
        return

    units_seen = 0
    for example in examples:
        example_units = example.effective_units()
        if units_seen + example_units > max_eval_units:
            break
        units_seen += example_units
        yield example
