# -*- coding: utf-8 -*-

from pathlib import Path

from wintermute.ml.runs import RunManifest, RunStore
from wintermute.tools.params import get_optional_str, get_str
from wintermute.ml.training.config import TrainingConfig
from wintermute.ml.training.workspace import TrainingWorkspace
from wintermute.tools.files import yaml_load
from wintermute.tools.logging import console_log

def run_new(output_root: str, training_config_file: str) -> str:
    run_id = _create_training_run(output_root, training_config_file)
    _run_fit(Path(output_root), run_id, source_operation="run-new")
    return run_id


def run_resume(output_root: str, run_id: str, checkpoint: str = "last") -> None:
    root = Path(output_root)
    run = RunStore(root).open(run_id, load_slack_config=True)
    if run.manifest.run_id != run_id:
        raise ValueError(
            f"run manifest id '{run.manifest.run_id}' does not match run store id '{run_id}'"
        )
    if run.checkpoints.selected(checkpoint, with_states=False) is None:
        raise ValueError(f"checkpoint '{checkpoint}' not found for run '{run_id}'")
    _run_fit(
        root,
        run_id,
        resume_checkpoint=checkpoint,
        source_operation="resume",
    )


def run_fork(
    output_root: str,
    source_run_id: str,
    destination_run_id: str,
    checkpoint: str = "last",
) -> str:
    return RunStore(output_root).fork(source_run_id, destination_run_id, checkpoint)


def _create_training_run(output_root: str, training_config_file: str) -> str:
    root = Path(output_root)
    cfg_path = Path(training_config_file)
    parameters = yaml_load(cfg_path.parent.as_posix(), cfg_path.name)

    config = TrainingConfig.from_dict(_load_training_configs(parameters))
    notes = get_str(parameters, "notes", "") or ""
    run_id = get_optional_str(parameters, "run_id") or cfg_path.stem

    runs = RunStore(root)
    run_id = runs.allocate_run_id(run_id)

    runs.save(
        RunManifest(
            run_id=run_id,
            notes=str(notes),
            config=config,
        )
    )
    return run_id


def _load_training_configs(parameters: object) -> dict:
    if not isinstance(parameters, dict):
        raise TypeError("training config root must be a YAML mapping")

    config_payload = parameters.get("configuration")
    if not isinstance(config_payload, dict):
        raise ValueError("training config YAML must contain a 'configuration' mapping")
    return config_payload


def _run_fit(
    root: Path,
    run_id: str,
    *,
    resume_checkpoint: str = "last",
    source_operation: str = "run-new",
) -> None:
    with TrainingWorkspace.open(root, run_id) as workspace:
        if workspace.run.has_saved_model():
            raise ValueError(
                f"final model already exists for run '{run_id}' in '{workspace.run.model_path.as_posix()}'"
            )

        workspace.capture_source(
            operation=source_operation,
            checkpoint_selector=(
                resume_checkpoint if source_operation == "resume" else None
            ),
        )

        trainer = workspace.create_trainer(resume_checkpoint=resume_checkpoint)
        model = trainer.train()

        console_log("training workspace", "saving model")
        workspace.save_model(
            model,
            trainer.task_wrapper,
        )
        console_log("training workspace", "model saved - end of training")        
