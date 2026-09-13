from __future__ import annotations

from wintermute.data.transform.base import RecordTransform
from wintermute.data.transform.config import RecordTransformConfig
from wintermute.data.transform.implementations.arithmetic_augmentation import (
    ArithmeticTemplateAugmentationRecordTransform,
    EquationAnswerArithmeticAugmentationRecordTransform,
)
from wintermute.data.transform.implementations.chat_to_sft import ChatToSftRecordTransform
from wintermute.data.transform.implementations.composite import FanOutTransform, SequentialTransform
from wintermute.data.transform.implementations.field_mapping import FieldMappingRecordTransform
from wintermute.data.transform.implementations.helpers import ensure_allowed_params
from wintermute.data.transform.implementations.lookup_retrieval import LookupRetrievalRecordTransform
from wintermute.data.transform.implementations.split_qa_pairs import SplitQaPairsRecordTransform
from wintermute.data.transform.implementations.to_sft import ToSftRecordTransform


def build_record_transform(config: RecordTransformConfig) -> RecordTransform:
    params = dict(config.params)
    if config.kind == FanOutTransform.KIND:
        ensure_allowed_params(params, set(), kind=FanOutTransform.KIND)
        return FanOutTransform(
            build_record_transform(child)
            for child in config.children
        )
    if config.kind == SequentialTransform.KIND:
        ensure_allowed_params(params, set(), kind=SequentialTransform.KIND)
        return SequentialTransform(
            build_record_transform(child)
            for child in config.children
        )
    if config.children:
        raise ValueError(f"{config.kind} does not support child transforms")
    if config.kind == FieldMappingRecordTransform.KIND:
        ensure_allowed_params(params, {"fields", "task"}, kind=FieldMappingRecordTransform.KIND)
        return FieldMappingRecordTransform(fields=params["fields"], task=params.get("task"))
    if config.kind == ChatToSftRecordTransform.KIND:
        return ChatToSftRecordTransform(params)
    if config.kind == ToSftRecordTransform.KIND:
        return ToSftRecordTransform(params)
    if config.kind == ArithmeticTemplateAugmentationRecordTransform.KIND:
        return ArithmeticTemplateAugmentationRecordTransform(params)
    if config.kind == EquationAnswerArithmeticAugmentationRecordTransform.KIND:
        return EquationAnswerArithmeticAugmentationRecordTransform(params)
    if config.kind == LookupRetrievalRecordTransform.KIND:
        return LookupRetrievalRecordTransform.from_params(params)
    if config.kind == SplitQaPairsRecordTransform.KIND:
        return SplitQaPairsRecordTransform(params)
    raise ValueError(f"Unknown record transform kind: {config.kind}")
