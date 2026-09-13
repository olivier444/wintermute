from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any
import json
import hashlib
from wintermute.tools.files import json_save, json_load

@dataclass(frozen=True)
class ModelSpecification:
    builder_class: str
    params: Dict[str, Any]

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    
    def save(self, directory: str, file_name: str) -> None:
        json_save(self, directory, file_name)

    @classmethod
    def from_dict(cls, data: dict) -> ModelSpecification:
        payload = dict(data)
        payload.pop("name", None)
        return cls(**payload)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, directory: str, file_name: str) -> ModelSpecification:
        return cls.from_dict(json_load(directory, file_name))
    
