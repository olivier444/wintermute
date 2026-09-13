from __future__ import annotations

import re
import shutil
from pathlib import Path
from uuid import uuid4

from wintermute.tools.store import AbstractStore
from wintermute.tools.files import json_save

from wintermute.ml.runs.checkpoints import (
    CHECKPOINT_DIRNAME,
    CHECKPOINT_LAST_DIRNAME,
    CheckPoint,
    RunCheckpoints,
    _CheckPointStore,
)
from wintermute.ml.runs.manifest import RUN_DESC_FILENAME, RunManifest
from wintermute.ml.runs.run import MODEL_DIRNAME, Run


RUN_STORE_DIRNAME = "run_store"
FORK_METADATA_FILENAME = "fork.json"
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class _ManifestStore(AbstractStore):
    def load(
        self,
        run_id: str,
        *,
        load_slack_config: bool = False,
    ) -> RunManifest:
        return RunManifest.load(
            self.get_object_dir(run_id),
            RUN_DESC_FILENAME,
            load_slack_config=load_slack_config,
        )

    def save(self, manifest: RunManifest) -> None:
        manifest.save(self.get_object_dir(manifest.run_id, ensure_exist=True), RUN_DESC_FILENAME)


class RunStore:
    def __init__(
        self,
        root: str | Path,
        *,
        checkpoint_history: int = 6,
    ) -> None:
        self.root = Path(root)
        self._manifests = _ManifestStore(self.root / RUN_STORE_DIRNAME)
        self._checkpoint_store = _CheckPointStore(
            self.root / RUN_STORE_DIRNAME,
            max_history=max(1, int(checkpoint_history)),
        )

    def list(self) -> list[str]:
        return self._manifests.list()

    def allocate_run_id(self, candidate: str) -> str:
        prefix = candidate.strip()
        run_id = prefix
        existing = set(self.list())
        next_suffix = 1

        while run_id in existing:
            next_suffix += 1
            run_id = f"{prefix}-{next_suffix:03d}"

        return run_id

    def save(self, manifest: RunManifest) -> None:
        self._manifests.save(manifest)

    def fork(
        self,
        source_run_id: str,
        destination_run_id: str,
        checkpoint: str = "last",
    ) -> str:
        source = self.open(source_run_id)
        if source.manifest.run_id != source_run_id:
            raise ValueError(
                f"run manifest id '{source.manifest.run_id}' does not match run store id '{source_run_id}'"
            )

        destination = _validate_run_id(destination_run_id)
        destination_path = self._run_path(destination, silent=True)
        if destination_path.exists():
            raise ValueError(f"destination run '{destination}' already exists")

        checkpoint_selector = checkpoint.strip()
        source_checkpoint = source.checkpoints.selected(checkpoint_selector, with_states=True)
        if source_checkpoint is None:
            raise ValueError(f"checkpoint '{checkpoint}' not found for run '{source_run_id}'")

        temporary_path = destination_path.parent / f".{destination}.fork-{uuid4().hex}"
        try:
            shutil.copytree(source.path, temporary_path)
            self._prepare_fork_directory(temporary_path, source_checkpoint)
            RunManifest(
                run_id=destination,
                notes=source.manifest.notes,
                config=source.manifest.config,
            ).save(temporary_path)
            json_save(
                {
                    "source_run_id": source_run_id,
                    "checkpoint_selector": checkpoint_selector,
                    "checkpoint_id": source_checkpoint.checkpoint_id,
                },
                temporary_path.as_posix(),
                FORK_METADATA_FILENAME,
            )
            temporary_path.rename(destination_path)
        finally:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)

        return destination

    def open(self, run_id: str, *, load_slack_config: bool = False) -> Run:
        manifest = self._manifests.load(
            run_id,
            load_slack_config=load_slack_config,
        )
        run_path = self._run_path(run_id)
        return Run(
            root=self.root,
            run_id=run_id,
            path=run_path,
            manifest=manifest,
            checkpoints=self.checkpoints_for(run_id),
        )

    def try_open(self, run_id: str, *, load_slack_config: bool = False) -> Run | None:
        run_path = self._run_path(run_id, silent=True)
        if not run_path.exists():
            return None

        try:
            manifest = self._manifests.load(
                run_id,
                load_slack_config=load_slack_config,
            )
        except Exception:
            return None

        return Run(
            root=self.root,
            run_id=run_id,
            path=run_path,
            manifest=manifest,
            checkpoints=self.checkpoints_for(run_id),
        )

    def checkpoints_for(self, run_id: str) -> RunCheckpoints:
        return self._checkpoint_store.for_run(run_id)

    def _run_path(
        self,
        run_id: str,
        *,
        silent: bool = False,
    ) -> Path:
        return Path(self._manifests.get_object_dir(run_id, silent=silent))

    @staticmethod
    def _prepare_fork_directory(temporary_path: Path, checkpoint: CheckPoint) -> None:
        shutil.rmtree(temporary_path / CHECKPOINT_DIRNAME, ignore_errors=True)
        shutil.rmtree(temporary_path / MODEL_DIRNAME, ignore_errors=True)
        (temporary_path / "slack.json").unlink(missing_ok=True)
        for control_file in (".params", ".pause", ".stop"):
            (temporary_path / control_file).unlink(missing_ok=True)

        checkpoint.save((temporary_path / CHECKPOINT_DIRNAME / f"{CHECKPOINT_LAST_DIRNAME}00000").as_posix())


def _validate_run_id(run_id: str) -> str:
    cleaned = run_id.strip()
    if not _RUN_ID_RE.fullmatch(cleaned):
        raise ValueError(
            "destination run id must use letters, digits, dots, underscores, or hyphens, "
            "and must start with a letter or digit"
        )
    return cleaned
