from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from wintermute.data.constants import DIR_EXTERNAL_MODELS
from wintermute.ml.evaluation.lm_eval.config import LmEvalConfig
from wintermute.tools.logging import console_log


@dataclass(frozen=True)
class LmEvalHarnessResult:
    results: Mapping[str, Any]
    cache_path: Path


class LmEvalHarness:
    """Own the lm-eval process environment and invoke its public API."""

    def __init__(self, output_root: str | Path) -> None:
        self.cache_path = self._configure_cache(Path(output_root))

    def evaluate(self, adapter: Any, config: LmEvalConfig) -> LmEvalHarnessResult:
        console_log(
            "eval",
            "starting lm-eval tasks: " + ", ".join(map(str, self._normalize_tasks(config.values["tasks"]))),
        )
        console_log("eval", f"lm-eval cache: {self.cache_path.as_posix()}")
        results = self._run(adapter, config.harness_values, device=config.device)
        console_log(
            "eval",
            "completed lm-eval tasks: " + ", ".join(sorted(results.get("results", {}))),
        )
        return LmEvalHarnessResult(results=results, cache_path=self.cache_path)

    @staticmethod
    def _configure_cache(output_root: Path) -> Path:
        cache_root = output_root / DIR_EXTERNAL_MODELS / "lm_eval"
        datasets_cache = cache_root / "datasets"
        downloads_cache = datasets_cache / "downloads"
        extracted_cache = downloads_cache / "extracted"
        hub_cache = cache_root / "hub"
        for path in (datasets_cache, downloads_cache, extracted_cache, hub_cache):
            path.mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"] = cache_root.as_posix()
        os.environ["HF_HUB_CACHE"] = hub_cache.as_posix()
        os.environ["HF_DATASETS_CACHE"] = datasets_cache.as_posix()
        os.environ["HF_DATASETS_DOWNLOADED_DATASETS_PATH"] = downloads_cache.as_posix()
        os.environ["HF_DATASETS_EXTRACTED_DATASETS_PATH"] = extracted_cache.as_posix()

        from datasets import config as datasets_config
        from huggingface_hub import constants as huggingface_constants

        datasets_config.HF_DATASETS_CACHE = datasets_cache
        datasets_config.DOWNLOADED_DATASETS_PATH = downloads_cache
        datasets_config.EXTRACTED_DATASETS_PATH = extracted_cache
        huggingface_constants.HF_HOME = cache_root.as_posix()
        huggingface_constants.HF_HUB_CACHE = hub_cache.as_posix()
        huggingface_constants.HUGGINGFACE_HUB_CACHE = hub_cache.as_posix()
        huggingface_constants.HF_ASSETS_CACHE = (cache_root / "assets").as_posix()
        return cache_root

    @classmethod
    def _run(
        cls,
        adapter: Any,
        raw_config: Mapping[str, Any],
        *,
        device: Any,
    ) -> Mapping[str, Any]:
        try:
            import lm_eval  # type: ignore[import-not-found]
            from lm_eval.tasks import TaskManager  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "lm-evaluation-harness is not installed; install the Wintermute project dependencies"
            ) from exc

        config = dict(raw_config)
        tasks = cls._normalize_tasks(config.pop("tasks"))
        include_path = config.pop("include_path", None)
        config.pop("config", None)
        config.pop("output_path", None)
        config.pop("show_config", None)
        configured_model = config.pop("model", None)
        configured_model_args = config.pop("model_args", None)
        config.pop("device", None)
        config.setdefault("batch_size", 1)
        config.setdefault("fewshot_as_multiturn", False)
        config.setdefault("log_samples", False)
        if configured_model is not None or configured_model_args:
            console_log("eval", "ignoring lm-eval model/model_args; using the selected evaluation target")

        for unsupported_key in ("wandb_args", "wandb_config_args", "hf_hub_log_args", "trackio_args"):
            if config.pop(unsupported_key, None):
                raise ValueError(
                    f"lm-eval config option '{unsupported_key}' is not supported by Wintermute eval"
                )

        if bool(config.pop("trust_remote_code", False)):
            import datasets

            datasets_config = getattr(datasets, "config")
            datasets_config.HF_DATASETS_TRUST_REMOTE_CODE = True

        seed = config.pop("seed", None)
        if seed is not None:
            random_seed, numpy_seed, torch_seed, fewshot_seed = cls._normalize_seeds(seed)
            config.update(
                random_seed=random_seed,
                numpy_random_seed=numpy_seed,
                torch_random_seed=torch_seed,
                fewshot_random_seed=fewshot_seed,
            )

        samples = config.get("samples")
        if isinstance(samples, str):
            samples_path = Path(samples)
            config["samples"] = (
                json.loads(samples_path.read_text(encoding="utf-8"))
                if samples_path.is_file()
                else json.loads(samples)
            )

        cls._normalize_request_cache(config)
        metadata = config.get("metadata")
        task_manager = TaskManager(
            include_path=include_path,
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )
        config.update(model=adapter, tasks=tasks, task_manager=task_manager, device=str(device))

        supported = set(inspect.signature(lm_eval.simple_evaluate).parameters)
        unsupported = sorted(key for key in config if key not in supported)
        if unsupported:
            raise ValueError(f"unsupported lm-eval config option(s): {', '.join(unsupported)}")

        results = lm_eval.simple_evaluate(**config)
        if results is None:
            raise RuntimeError("lm-eval did not return evaluation results")
        return results

    @staticmethod
    def _normalize_tasks(raw_tasks: Any) -> list[Any]:
        if isinstance(raw_tasks, str):
            return [task.strip() for task in raw_tasks.split(",") if task.strip()]
        if isinstance(raw_tasks, list) and raw_tasks:
            return list(raw_tasks)
        raise TypeError("lm-eval config 'tasks' must be a task name or non-empty list")

    @staticmethod
    def _normalize_seeds(raw_seed: Any) -> tuple[int | None, int | None, int | None, int | None]:
        values = list(raw_seed) if isinstance(raw_seed, (list, tuple)) else [raw_seed]
        if len(values) == 1:
            values *= 4
        if len(values) != 4:
            raise ValueError("lm-eval seed must be one value or four values")
        return tuple(None if value is None else int(value) for value in values)  # type: ignore[return-value]

    @staticmethod
    def _normalize_request_cache(config: dict[str, Any]) -> None:
        cache_requests = config.get("cache_requests")
        if isinstance(cache_requests, Mapping):
            config.pop("cache_requests")
            allowed = {"cache_requests", "rewrite_requests_cache", "delete_requests_cache"}
            unsupported = sorted(set(cache_requests) - allowed)
            if unsupported:
                raise ValueError("unsupported lm-eval cache_requests option(s): " + ", ".join(unsupported))
            config.update(cache_requests)
            return
        if not isinstance(cache_requests, str):
            return
        mode = cache_requests.strip().lower()
        if mode == "true":
            config["cache_requests"] = True
        elif mode == "refresh":
            config["cache_requests"] = True
            config["rewrite_requests_cache"] = True
        elif mode == "delete":
            config["cache_requests"] = False
            config["delete_requests_cache"] = True
        else:
            raise ValueError("lm-eval cache_requests must be true, refresh, or delete")
