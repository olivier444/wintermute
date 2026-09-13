from __future__ import annotations

from collections.abc import Iterable

from wintermute.data.record import DataRecord
from wintermute.data.transform.base import RecordTransform
from wintermute.data.transform.context import TransformContext


def _apply_transform(
    transform: RecordTransform,
    records: Iterable[DataRecord],
    *,
    context: TransformContext,
) -> Iterable[DataRecord]:
    for record in records:
        yield from transform.transform(record, context=context)


class SequentialTransform(RecordTransform):
    """Apply child transforms as a pipeline.

    Every output of one child is fed to the next child, so a child that drops
    or expands records affects all later stages.  Use it to build one ordered
    transformation path.
    """

    KIND = "sequential"

    def __init__(self, transforms: Iterable[RecordTransform]) -> None:
        self.transforms = tuple(transforms)

    def transform(
        self,
        record: DataRecord,
        *,
        context: TransformContext,
    ) -> Iterable[DataRecord]:
        records: Iterable[DataRecord] = (record,)
        for transform in self.transforms:
            records = _apply_transform(
                transform,
                records,
                context=context,
            )
        return records


class FanOutTransform(RecordTransform):
    """Apply each child transform independently to the same input record.

    Outputs from every branch are combined rather than passed between branches.
    When several branches exist, output identifiers gain branch and position
    suffixes to remain unique.
    """

    KIND = "fan_out"

    def __init__(self, transforms: Iterable[RecordTransform]) -> None:
        self.transforms = tuple(transforms)

    def transform(
        self,
        record: DataRecord,
        *,
        context: TransformContext,
    ) -> Iterable[DataRecord]:
        add_transform_suffix = len(self.transforms) > 1
        for transform_index, transform in enumerate(self.transforms):
            for output_index, output_record in enumerate(
                transform.transform(record, context=context)
            ):
                if add_transform_suffix:
                    yield DataRecord(
                        record_id=(
                            f"{output_record.record_id}.transform_"
                            f"{transform_index}.{output_index}"
                        ),
                        fields=output_record.fields,
                    )
                else:
                    yield output_record
