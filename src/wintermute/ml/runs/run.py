from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from wintermute.ml.models.factory import ModelFactory
from wintermute.ml.models.base import BaseModel
from wintermute.ml.models.specifications import ModelSpecification
from wintermute.ml.runs.checkpoints import CheckPoint, RunCheckpoints
from wintermute.ml.runs.manifest import RunManifest
from wintermute.ml.training.config import TrainingConfig
from wintermute.tools.files import json_save
from wintermute.tools.logging import console_log
from wintermute.tools.misc import utc_now_unix_ms
from wintermute.tools.model import count_parameters


SPEC_FILENAME = "specification.json"
METADATA_FILENAME = "metadata.json"
WEIGHTS_FILENAME = "weights.pt"
LEGACY_WEIGHTS_FILENAME = "weights"
MODEL_DIRNAME = "model"


@dataclass(frozen=True)
class Run:
    root: Path
    run_id: str
    path: Path
    manifest: RunManifest
    checkpoints: RunCheckpoints

    @property
    def config(self) -> TrainingConfig:
        return self.manifest.config

    @property
    def notes(self) -> str:
        return self.manifest.notes

    @property
    def spec(self) -> ModelSpecification:
        return self.manifest.config.model_config

    @property
    def model_path(self) -> Path:
        return self.path / MODEL_DIRNAME

    def model_artifact_path(
        self,
        directory_name: str = MODEL_DIRNAME,
    ) -> Path:
        return self.path / directory_name

    def build_model(self) -> BaseModel:
        return ModelFactory().build_model_from_spec(self.spec)

    def build_loaded_model(
        self,
        *,
        device: str | torch.device = "cpu",
        allow_checkpoint_fallback: bool = True,
    ) -> BaseModel:
        model = self.build_model()
        if allow_checkpoint_fallback:
            if not self.try_load_into(model, device=device):
                raise ValueError(f"failed to load model for run '{self.run_id}'")
        else:
            self.load_saved_model_into(model, device=device)
        return model

    def has_saved_model(self, directory_name: str = MODEL_DIRNAME) -> bool:
        model_path = self.model_artifact_path(directory_name)
        return model_path.exists() and any(model_path.iterdir())

    def save_model(
        self,
        model: nn.Module,
        *,
        overwrite: bool = False,
        directory_name: str = MODEL_DIRNAME,
        code_version: Optional[str] = None,
        notes: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        model_path = self.model_artifact_path(directory_name)
        if self.has_saved_model(directory_name) and not overwrite:
            raise ValueError(f"model directory already exists for run '{self.run_id}': {model_path}")

        model_path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": self.run_id if directory_name == MODEL_DIRNAME else f"{self.run_id}/{directory_name}",
            "created_at_utc": utc_now_unix_ms(),
            "code_version": code_version,
            "notes": notes,
            "params": count_parameters(model),
            "fingerprint": self.spec.fingerprint(),
            "extra": extra_meta or {},
        }

        json_save(metadata, model_path.as_posix(), METADATA_FILENAME)
        self.spec.save(model_path.as_posix(), SPEC_FILENAME)
        torch.save(model.state_dict(), model_path / WEIGHTS_FILENAME)

    def _load_saved_model_state(
        self,
        *,
        device: str | torch.device = "cpu",
    ) -> Optional[Dict[str, Any]]:
        for file_name in (WEIGHTS_FILENAME, LEGACY_WEIGHTS_FILENAME):
            weights_path = self.model_path / file_name
            if weights_path.exists():
                return torch.load(weights_path, map_location=device)
        return None

    def try_load_into(
        self,
        model: nn.Module,
        *,
        device: str | torch.device = "cpu",
    ) -> bool:
        try:
            self.load_saved_model_into(model, device=device)
            return True
        except Exception as exc:
            console_log("run-loader", f"failed to load model: {exc}")
        return self._try_load_checkpoint_into(model)

    def load_saved_model_into(
        self,
        model: nn.Module,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        model_state = self._load_saved_model_state(device=device)
        if model_state is None:
            raise ValueError(f"saved model not found for run '{self.run_id}' in {self.model_path}")

        console_log("run-loader", f"loading saved model from run-id {self.run_id}")
        model.load_state_dict(model_state)

    def load_checkpoint_into(
        self,
        model: nn.Module,
        checkpoint_selector: str,
    ) -> CheckPoint:
        checkpoint = self.checkpoints.selected(checkpoint_selector, with_states=True)
        if checkpoint is None:
            raise ValueError(
                f"checkpoint '{checkpoint_selector}' not found for reference run '{self.run_id}'"
            )

        model_state = checkpoint.states.get("model")
        if model_state is None:
            raise ValueError(
                f"checkpoint '{checkpoint.checkpoint_id}' for reference run '{self.run_id}' has no model state"
            )

        console_log(
            "run-loader",
            f"loading model from run-id {self.run_id} and checkpoint {checkpoint.checkpoint_id} "
            f"(selector='{checkpoint_selector}')",
        )
        model.load_state_dict(model_state)
        return checkpoint

    def _try_load_checkpoint_into(
        self,
        model: nn.Module,
    ) -> bool:
        try:
            self.load_checkpoint_into(model, "last")
        except Exception as exc:
            console_log("run-loader", f"failed to read checkpoint for reference run {self.run_id}: {exc}")
            return False
        return True
