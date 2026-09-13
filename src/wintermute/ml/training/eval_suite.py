from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from wintermute.data.iterate.dataclasses import TrainingViewConfig
from wintermute.ml.tasks.factory import TaskWrapper


@dataclass
class RuntimeEvalSuite:
    name: str
    task_wrapper: TaskWrapper
    view_config: TrainingViewConfig
    batch_size: int
    max_eval_units: Optional[int] = None
    is_primary: bool = False
    evaluation_modulus: int = 1
    _macro_steps_since_evaluation: int = 0

    def is_evaluation_due(self) -> bool:
        self._macro_steps_since_evaluation += 1
        if self._macro_steps_since_evaluation < self.evaluation_modulus:
            return False

        self._macro_steps_since_evaluation = 0
        return True

    def reconfigure_modulus(self, modulus: int) -> bool:
        if modulus == self.evaluation_modulus:
            return False

        self.evaluation_modulus = modulus
        self._macro_steps_since_evaluation = 0
        return True


@dataclass(frozen=True)
class EvalStepResult:
    metrics_by_suite: Dict[str, Dict[str, float]]
    primary_suite_name: str

    @property
    def primary_metrics(self) -> Dict[str, float]:
        return self.metrics_by_suite[self.primary_suite_name]

    @property
    def primary_loss(self) -> float:
        return float(self.primary_metrics["loss"])
