from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

from wintermute.tools.params import get_optional_mapping


@dataclass
class TaskSpecification:
    wrapper_class: str
    params: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSpecification":
        d = dict(data)
        params = get_optional_mapping(d, "params")
        return cls(
            wrapper_class=d["wrapper_class"],
            params=dict(params) if params is not None else {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
