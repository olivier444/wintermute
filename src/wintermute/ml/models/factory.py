from __future__ import annotations

from wintermute.ml.models.base import BaseModel
from wintermute.ml.models.specifications import ModelSpecification
from wintermute.tools.misc import resolve_class_from_package


class ModelFactory:
    def __init__(self) -> None:
        return
    
        
    def build_model_from_spec(self, specification: ModelSpecification) -> BaseModel:
        builder_cls = resolve_class_from_package(
            package_name="wintermute.ml.models.builders",
            class_name=specification.builder_class,
            expected_base_class=BaseModel
        )

        try:
            return builder_cls(specification)
        except TypeError as exc:
            raise TypeError(
                f"Builder '{specification.builder_class}' must be instantiable with ModelSpecification."
            ) from exc        
