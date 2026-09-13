from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from wintermute.data.constants import DIR_EXTERNAL_MODELS, DIR_HUGGING_FACE_MODELS
from wintermute.ml.inference.generator import Generator
from wintermute.ml.runs import Run, RunStore
from wintermute.ml.tasks.factory import TaskWrapper, TaskWrapperFactory
from wintermute.ml.training.task_specification import TaskSpecification


class InferenceWorkspace:
    def __init__(
        self,
        run: Run,
    ) -> None:
        self.run = run
        self.run_id = run.run_id
        self.root_path = run.root
        self.run_dir = run.path
        self.manifest = run.manifest
        self.checkpoints = run.checkpoints
        self._wrapper_factory = TaskWrapperFactory()

    @property
    def config(self):
        return self.run.config

    def create_components(
        self,
        *,
        device_override: str | torch.device | None = None,
        external_model_name: str | None = None,
    ) -> tuple[TaskWrapper, nn.Module]:
        inference_device = self._resolve_inference_device(device_override)
        bf16_enabled = bool(self.config.trainer_config.use_bf16 and inference_device.type == "cuda")
        if external_model_name:
            wrapper, model = self._create_external_components(
                inference_device=inference_device,
                bf16_enabled=bf16_enabled,
                external_model_name=external_model_name,
            )
        else:
            model = self.run.build_loaded_model()
            wrapper = self._wrapper_factory.build_wrapper(
                self.config.task_config,
                inference_device,
                bf16_enabled,
                self.root_path,
            )
        model.to(wrapper.device)
        return wrapper, model

    def create_generator(
        self,
        *,
        system_prompt: str | None = None,
        device_override: str | torch.device | None = None,
    ) -> Generator:
        task_wrapper, model = self.create_components(device_override=device_override)
        return Generator(
            task_wrapper=task_wrapper,
            model=model,
            system_prompt=system_prompt,
        )

    @classmethod
    def open(
        cls,
        root: Path,
        run_id: str,
        *,
        load_slack_config: bool = False,
    ) -> "InferenceWorkspace":
        runs = RunStore(root)
        return cls(runs.open(run_id, load_slack_config=load_slack_config))

    def close(self) -> None:
        return

    def __enter__(self) -> "InferenceWorkspace":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def _resolve_inference_device(
        self,
        device_override: str | torch.device | None,
    ) -> torch.device:
        if device_override is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(device_override, torch.device):
            resolved = device_override
        else:
            clean = device_override.strip().lower()
            if clean == "gpu":
                clean = "cuda"
            resolved = torch.device(clean)

        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested for inference, but no GPU is available")
        if resolved.type not in {"cpu", "cuda"}:
            raise ValueError("Inference device must be 'cpu' or 'cuda'")
        return resolved

    def _create_external_components(
        self,
        *,
        inference_device: torch.device,
        bf16_enabled: bool,
        external_model_name: str,
    ) -> tuple[TaskWrapper, nn.Module]:
        tokenizer_name = external_model_name.strip()
        task_params = dict(self.config.task_config.params)
        task_params.pop("tokenizer_id", None)
        task_params["hf_tokenizer_name"] = tokenizer_name
        task = TaskSpecification(
            wrapper_class=self.config.task_config.wrapper_class,
            params=task_params,
        )
        wrapper = self._wrapper_factory.build_wrapper(
            task,
            inference_device,
            bf16_enabled,
            self.root_path,
        )

        cache_dir = self.root_path / DIR_EXTERNAL_MODELS / DIR_HUGGING_FACE_MODELS
        print(
            f"[inference workspace] loading external model={external_model_name} "
            f"tokenizer={tokenizer_name}"
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            external_model_name,
            cache_dir=str(cache_dir),
        )
        model = cast(nn.Module, hf_model)
        return wrapper, model
