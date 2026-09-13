from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from wintermute.tools.params import normalize_kind_params_payload


@dataclass(frozen=True)
class RecordTransformConfig:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    children: tuple[RecordTransformConfig, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.kind or not self.kind.strip():
            raise ValueError("RecordTransformConfig.kind must be a non-empty string")

    @classmethod
    def from_dict(
        cls,
        payload: str | Mapping[str, Any],
    ) -> RecordTransformConfig:
        kind, params = normalize_kind_params_payload(
            payload,
            context="RecordTransformConfig",
        )
        raw_children = params.pop("children", ())
        if isinstance(raw_children, (str, bytes, bytearray)) or not isinstance(
            raw_children,
            Sequence,
        ):
            raise TypeError("RecordTransformConfig.children must be a sequence")
        return cls(
            kind=kind,
            params=params,
            children=tuple(cls.from_dict(child) for child in raw_children),
        )

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.params)
        data["kind"] = self.kind
        if self.children:
            data["children"] = [child.to_dict() for child in self.children]
        return data
