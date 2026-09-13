from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, cast

import torch.nn as nn
from transformers import AutoModelForCausalLM

from wintermute.data.constants import DIR_EXTERNAL_MODELS, DIR_HUGGING_FACE_MODELS
from wintermute.data.iterate.dataclasses import TrainingViewConfig
from wintermute.ml.runs import Run, RunStore
from wintermute.ml.tasks.baseline import BaselineEvaluator, BaselineSuite
from wintermute.ml.tasks.factory import TaskWrapper
from wintermute.tools.logging import console_log
from wintermute.tools.params import get_int


class CausalLmBaselineSuite(BaselineSuite):
    def __init__(
            self,
            wrapper_builder: Callable[[str], TaskWrapper],
            run_wrapper_builder: Callable[[Run], TaskWrapper],
            baseline_config: Mapping[str, Any],
            output_root: Path
    ):
        self.output_root = output_root
        batch_size = get_int(baseline_config, "batch_size", 1)

        self.evaluators = [
            self._build_hf_evaluator(
                label,
                model_name,
                wrapper_builder(model_name),
                batch_size,
            )
            for label, model_name in self._external_models(baseline_config).items()
        ]
        self.evaluators.extend(
            self._build_run_evaluator(
                label,
                run_id,
                run_wrapper_builder,
                batch_size,
            )
            for label, run_id in self._run_refs(baseline_config).items()
        )


    def _build_hf_evaluator(
        self,
        label: str,
        model_name: str,
        wrapper: TaskWrapper,
        batch_size: int,
    ) -> BaselineEvaluator:
        console_log("baseline", f"loading {model_name}")
        cache_dir = self.output_root / DIR_EXTERNAL_MODELS / DIR_HUGGING_FACE_MODELS
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache_dir))
        model = cast(nn.Module, hf_model)
        return BaselineEvaluator(
            label,
            model,
            wrapper,
            batch_size,
        )


    def _build_run_evaluator(
        self,
        label: str,
        run_id: str,
        run_wrapper_builder: Callable[[Run], TaskWrapper],
        batch_size: int,
    ) -> BaselineEvaluator:
        console_log("baseline", f"loading run baseline {run_id} (saved model)")
        run = RunStore(self.output_root).open(run_id, load_slack_config=False)
        model = run.build_loaded_model(allow_checkpoint_fallback=False)
        return BaselineEvaluator(
            label,
            model,
            run_wrapper_builder(run),
            batch_size,
        )


    def _external_models(self, baseline_config: Mapping[str, Any]) -> Dict[str, str]:
        raw_models = baseline_config.get("external_models")
        if raw_models is None:
            return {}

        if isinstance(raw_models, Mapping):
            return {str(label): str(model_name) for label, model_name in raw_models.items()}

        if isinstance(raw_models, Sequence) and not isinstance(raw_models, (str, bytes)):
            models: Dict[str, str] = {}
            for model_name in raw_models:
                model_text = str(model_name)
                models[self._label_from_model_name(model_text)] = model_text
            return models

        raise TypeError("'baseline_config.external_models' must be a mapping or a sequence")


    def _run_refs(self, baseline_config: Mapping[str, Any]) -> Dict[str, str]:
        raw_runs = baseline_config.get("run_ids")
        if raw_runs is None:
            raw_runs = baseline_config.get("comparison_run_ids")
        if raw_runs is None:
            return {}

        if isinstance(raw_runs, Mapping):
            return {str(label): str(run_id) for label, run_id in raw_runs.items()}

        if isinstance(raw_runs, Sequence) and not isinstance(raw_runs, (str, bytes)):
            return {f"run_{run_id}": str(run_id) for run_id in raw_runs}

        raise TypeError("'baseline_config.run_ids' must be a mapping or a sequence")


    def _label_from_model_name(self, model_name: str) -> str:
        label = model_name.rsplit("/", 1)[-1].strip().lower()
        return label.replace("-", "_").replace(".", "_")


    def compute_eval_metrics(
        self,
        view_config: TrainingViewConfig
    ) -> Dict[str, Dict[str, float]]:
        return {
            evaluator.name: evaluator.compute_eval_metrics(
                view_config,
                str(self.output_root),
            )
            for evaluator in self.evaluators
        }

    def benchmark_evaluators(self) -> Sequence[BaselineEvaluator]:
        return tuple(self.evaluators)
