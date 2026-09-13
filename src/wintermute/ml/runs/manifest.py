from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from wintermute.ml.training.config import TrainingConfig
from wintermute.tools.files import json_load, json_save, yaml_load, yaml_save
from wintermute.tools.logging import console_log


RUN_DESC_FILENAME = "description.yaml"
LEGACY_RUN_DESC_FILENAME = "description.json"
TRAINING_VIEW_AUDIT_FILENAME = "training_view.yaml"
LEGACY_TRAINING_VIEW_AUDIT_FILENAME = "training_view.json"
RUN_MANIFEST_VERSION = 6


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    notes: str
    config: TrainingConfig

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        load_slack_config: bool = True,
    ) -> "RunManifest":
        payload = _migrate_run_manifest(dict(data))
        config_payload = _pop_configuration_payload(payload)
        payload.pop("version", None)
        return cls(
            run_id=str(payload["run_id"]),
            notes=str(payload.get("notes", "")),
            config=TrainingConfig.from_dict(
                config_payload,
                load_slack_config=load_slack_config,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": RUN_MANIFEST_VERSION,
            "run_id": self.run_id,
            "notes": self.notes,
            "configuration": self.config.to_dict(),
        }

    def save(self, directory: str | Path, file_name: str = RUN_DESC_FILENAME) -> None:
        directory_path = Path(directory)
        if Path(file_name).suffix == ".json":
            json_save(self.to_dict(), directory_path.as_posix(), file_name)
        else:
            yaml_save(self.to_dict(), directory_path.as_posix(), file_name)
        yaml_save(
            self.config.training_view_config.to_dict(),
            directory_path.as_posix(),
            TRAINING_VIEW_AUDIT_FILENAME,
        )
        if file_name == RUN_DESC_FILENAME:
            for legacy_file_name in (
                LEGACY_RUN_DESC_FILENAME,
                LEGACY_TRAINING_VIEW_AUDIT_FILENAME,
            ):
                legacy_path = directory_path / legacy_file_name
                if legacy_path.is_file():
                    legacy_path.unlink()

    @classmethod
    def load(
        cls,
        directory: str | Path,
        file_name: str = RUN_DESC_FILENAME,
        *,
        load_slack_config: bool = True,
    ) -> "RunManifest":
        directory_path = Path(directory)
        if file_name == RUN_DESC_FILENAME:
            current_path = directory_path / RUN_DESC_FILENAME
            legacy_path = directory_path / LEGACY_RUN_DESC_FILENAME
            if current_path.is_file() and legacy_path.is_file():
                raise ValueError(
                    f"run contains both '{RUN_DESC_FILENAME}' and "
                    f"'{LEGACY_RUN_DESC_FILENAME}': {directory_path}"
                )
            if not current_path.is_file() and legacy_path.is_file():
                file_name = LEGACY_RUN_DESC_FILENAME

        payload = (
            json_load(directory_path.as_posix(), file_name)
            if Path(file_name).suffix == ".json"
            else yaml_load(directory_path.as_posix(), file_name)
        )
        return cls.from_dict(
            payload,
            load_slack_config=load_slack_config,
        )


def _migrate_run_manifest(payload: Dict[str, Any]) -> Dict[str, Any]:
    version = int(payload.get("version", 1))

    while version < RUN_MANIFEST_VERSION:
        if version == 1:
            _migrate_v1_to_v2(payload)
            version = 2
            continue
        if version == 2:
            _migrate_v2_to_v3(payload)
            version = 3
            continue
        if version == 3:
            _migrate_v3_to_v4(payload)
            version = 4
            continue
        if version == 4:
            _migrate_v4_to_v5(payload)
            version = 5
            continue
        if version == 5:
            _migrate_v5_to_v6(payload)
            version = 6
            continue
        raise ValueError(f"Unsupported run manifest version: {version}")

    payload["version"] = version
    return payload


def _migrate_v1_to_v2(payload: Dict[str, Any]) -> None:
    config_payload = _get_configuration_payload(payload)
    if not isinstance(config_payload, dict):
        return

    task = config_payload.get("task")
    if not isinstance(task, dict):
        return

    if str(task.get("wrapper_class", "")).strip() != "CausalLmTaskWrapper":
        return

    params = task.get("params")
    if params is None:
        params = {}
        task["params"] = params

    if not isinstance(params, dict):
        return

    if "assistant_format" in params:
        return

    params["assistant_format"] = "legacy"
    run_id = str(payload.get("run_id", "<unknown>"))
    console_log(
        "run-manifest",
        f"migrated legacy run manifest for run '{run_id}' to v2 with assistant_format='legacy'",
    )


def _migrate_v2_to_v3(payload: Dict[str, Any]) -> None:
    if "configs" in payload:
        return
    config_payload = payload.pop("configuration", None)
    if isinstance(config_payload, dict):
        payload["configs"] = config_payload


def _migrate_v3_to_v4(payload: Dict[str, Any]) -> None:
    if "configuration" in payload:
        return
    config_payload = payload.pop("configs", None)
    if isinstance(config_payload, dict):
        payload["configuration"] = config_payload


def _migrate_v4_to_v5(payload: Dict[str, Any]) -> None:
    config_payload = _get_configuration_payload(payload)
    if not isinstance(config_payload, dict):
        return

    task_configs: list[object] = [config_payload.get("task_config")]
    evaluations = config_payload.get("evaluation_configs")
    if isinstance(evaluations, list):
        task_configs.extend(
            evaluation.get("task_config")
            for evaluation in evaluations
            if isinstance(evaluation, dict)
        )

    run_id = str(payload.get("run_id", "<unknown>"))
    for task_config in task_configs:
        _migrate_legacy_causal_lm_task(task_config, run_id)


def _migrate_v5_to_v6(payload: Dict[str, Any]) -> None:
    # Training-view topology is intentionally not migrated from inline JSON.
    # A v5-shaped payload is loadable only when it already names Python presets.
    return


def _get_configuration_payload(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    config_payload = payload.get("configuration")
    return config_payload if isinstance(config_payload, dict) else None



def _migrate_legacy_causal_lm_task(task_config: object, run_id: str) -> None:
    if not isinstance(task_config, dict):
        return
    if str(task_config.get("wrapper_class", "")).strip() != "CausalLmTaskWrapper":
        return

    params = task_config.get("params")
    if params is None:
        params = {}
        task_config["params"] = params
    if not isinstance(params, dict) or "assistant_format" in params:
        return

    params["assistant_format"] = "legacy"
    console_log(
        "run-manifest",
        f"migrated run '{run_id}' task params with assistant_format='legacy'",
    )


def _pop_configuration_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    config_payload = _get_configuration_payload(payload)
    if not isinstance(config_payload, dict):
        raise ValueError("run manifest must contain a 'configuration' object")
    payload.pop("configuration", None)
    return config_payload
