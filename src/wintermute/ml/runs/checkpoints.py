from __future__ import annotations

import json
import re
import shutil
from copy import copy
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import torch

from wintermute.tools.files import json_load
from wintermute.tools.logging import console_log
from wintermute.tools.misc import dt_to_unix_ms, unix_ms_to_dt
from wintermute.tools.store import AbstractStore


@dataclass
class CheckPoint:
    checkpoint_id: str
    states: Dict[str, Any]
    global_step: int
    units_seen: int
    examples_seen: int
    created_at: datetime
    train_loss: float
    val_loss: float

    @classmethod
    def from_dict(cls, data: dict) -> "CheckPoint":
        payload = dict(data)
        payload["created_at"] = unix_ms_to_dt(payload["created_at"])
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = dt_to_unix_ms(payload["created_at"])
        return payload

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        desc = copy(self)
        desc.states = {}
        with (path / CHECKPOINT_DESC_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(desc.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)

        for key, state in self.states.items():
            torch.save(state, path / f"{CHECKPOINT_STATE_FILENAME}{key}.pt")

    @classmethod
    def load(
        cls,
        directory: Optional[str],
        with_states: bool = False,
        map_location: str = "cpu",
    ) -> Optional["CheckPoint"]:
        if directory is None:
            return None

        dir_path = Path(directory)
        if not dir_path.exists():
            return None

        result = cls.from_dict(json_load(directory, CHECKPOINT_DESC_FILENAME))

        if with_states:
            state_files = sorted(
                [
                    path
                    for path in dir_path.iterdir()
                    if path.is_file()
                    and path.name.startswith(CHECKPOINT_STATE_FILENAME)
                    and path.suffix == ".pt"
                ]
            )
            for state_file in state_files:
                bucket_name = state_file.stem[len(CHECKPOINT_STATE_FILENAME):]
                bucket_data = torch.load(state_file, map_location=map_location)
                result.states[bucket_name] = bucket_data

        return result


CHECKPOINT_DESC_FILENAME = "description.json"
CHECKPOINT_STATE_FILENAME = "state_"
CHECKPOINT_DIRNAME = "checkpoints"
CHECKPOINT_VAL_BEST_DIRNAME = "val_best"
CHECKPOINT_LAST_DIRNAME = "cp_"
CHECKPOINT_NAMED_DIRNAME = "named"
_NAMED_CHECKPOINT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class RunCheckpoints:
    run_id: str
    _store: "_CheckPointStore"

    def register(self, checkpoint: CheckPoint) -> None:
        self._store.register_checkpoint(self.run_id, checkpoint)

    def best_val(self, *, with_states: bool) -> Optional[CheckPoint]:
        return self._store.get_best_val_checkpoint(self.run_id, with_states=with_states)

    def last(self, *, with_states: bool) -> Optional[CheckPoint]:
        return self._store.get_last_checkpoint(self.run_id, with_states=with_states)

    def selected(self, name: str, *, with_states: bool) -> Optional[CheckPoint]:
        return self._store.get_selected_checkpoint(self.run_id, name, with_states=with_states)

    def resolve(self, selector: str, *, with_states: bool) -> Optional[ResolvedCheckpoint]:
        return self._store.resolve_selected_checkpoint(
            self.run_id,
            selector,
            with_states=with_states,
        )

    def clone_last(self, name: str) -> CheckPoint:
        return self._store.clone_last_checkpoint(self.run_id, name)


@dataclass(frozen=True)
class ResolvedCheckpoint:
    selector: str
    path: Path
    checkpoint: CheckPoint


class _CheckPointStore(AbstractStore):
    def __init__(
        self,
        root_dir: str | Path,
        max_history: int = 6,
    ) -> None:
        super().__init__(root_dir)
        self.max_history = max_history

    def _run_dir(self, run_id: str) -> Path:
        return self.root / run_id / CHECKPOINT_DIRNAME

    def _get_checkpoint_dir(
        self,
        run_id: str,
        bucket_id: str,
        ensure_exist: bool = False,
    ) -> Optional[str]:
        run_dir = self._run_dir(run_id)
        if ensure_exist:
            run_dir.mkdir(parents=True, exist_ok=True)
        elif not run_dir.exists():
            return None

        path = run_dir / bucket_id
        if ensure_exist:
            path.mkdir(parents=True, exist_ok=True)
        elif not path.exists():
            return None
        return path.as_posix()

    def _list_history_dirs(self, run_id: str) -> list[str]:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return []
        return sorted(
            [path.name for path in run_dir.iterdir() if path.is_dir() and path.name.startswith(CHECKPOINT_LAST_DIRNAME)]
        )

    def _store(self, run_id: str, bucket: str, checkpoint: CheckPoint) -> None:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        target_path = run_dir / bucket
        temp_path = run_dir / f".{bucket}.tmp-{uuid4().hex}"
        backup_path = run_dir / f".{bucket}.bak-{uuid4().hex}"

        try:
            checkpoint.save(temp_path.as_posix())
            if target_path.exists():
                target_path.rename(backup_path)
                try:
                    temp_path.rename(target_path)
                except Exception:
                    if backup_path.exists() and not target_path.exists():
                        backup_path.rename(target_path)
                    raise
                self._delete_path(backup_path)
            else:
                temp_path.rename(target_path)
        finally:
            self._delete_path(temp_path)
            self._delete_path(backup_path)

    def _delete(self, run_id: str, bucket: str) -> None:
        path = self._run_dir(run_id) / bucket
        self._delete_path(path)

    def _delete_path(self, path: str | Path) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            return
        for file_path in path_obj.iterdir():
            file_path.unlink()
        path_obj.rmdir()

    def register_checkpoint(self, run_id: str, checkpoint: CheckPoint) -> None:
        console_log(
            "checkpoint",
            f"storing checkpoint #{checkpoint.checkpoint_id} (eval-loss={checkpoint.val_loss})",
        )
        val_best_path = self._get_checkpoint_dir(run_id, CHECKPOINT_VAL_BEST_DIRNAME)
        val_best_existing = CheckPoint.load(val_best_path, with_states=False)
        if val_best_existing is None or val_best_existing.val_loss > checkpoint.val_loss:
            self._store(run_id, CHECKPOINT_VAL_BEST_DIRNAME, checkpoint)

        history_dirs = self._list_history_dirs(run_id)
        while len(history_dirs) >= self.max_history:
            self._delete(run_id, history_dirs.pop(0))

        next_id = 0
        if history_dirs:
            next_id = int(history_dirs[-1][len(CHECKPOINT_LAST_DIRNAME):]) + 1
        history_bucket_name = f"{CHECKPOINT_LAST_DIRNAME}{next_id:05d}"
        self._store(run_id, history_bucket_name, checkpoint)

    def get_best_val_checkpoint(self, run_id: str, with_states: bool) -> Optional[CheckPoint]:
        return CheckPoint.load(
            self._get_checkpoint_dir(run_id, CHECKPOINT_VAL_BEST_DIRNAME),
            with_states=with_states,
        )

    def get_last_checkpoint(self, run_id: str, with_states: bool) -> Optional[CheckPoint]:
        checkpoint_path = self._get_last_checkpoint_dir(run_id)
        if checkpoint_path is None:
            return None
        return CheckPoint.load(checkpoint_path, with_states=with_states)

    def get_selected_checkpoint(
        self,
        run_id: str,
        name: str,
        *,
        with_states: bool,
    ) -> Optional[CheckPoint]:
        resolved = self.resolve_selected_checkpoint(
            run_id,
            name,
            with_states=with_states,
        )
        return resolved.checkpoint if resolved is not None else None

    def resolve_selected_checkpoint(
        self,
        run_id: str,
        name: str,
        *,
        with_states: bool,
    ) -> Optional[ResolvedCheckpoint]:
        selector = name.strip()
        checkpoint_path = self._get_selected_checkpoint_dir(run_id, selector)
        checkpoint = CheckPoint.load(checkpoint_path, with_states=with_states)
        if checkpoint is None or checkpoint_path is None:
            return None
        return ResolvedCheckpoint(
            selector=selector,
            path=Path(checkpoint_path),
            checkpoint=checkpoint,
        )

    def _get_selected_checkpoint_dir(
        self,
        run_id: str,
        selector: str,
    ) -> Optional[str]:
        if selector == "last":
            return self._get_last_checkpoint_dir(run_id)
        if selector == "best":
            return self._get_checkpoint_dir(run_id, CHECKPOINT_VAL_BEST_DIRNAME)

        checkpoint_name = _validate_named_checkpoint_name(selector)
        named_path = self._get_checkpoint_dir(
            run_id,
            f"{CHECKPOINT_NAMED_DIRNAME}/{checkpoint_name}",
        )
        if named_path is not None:
            return named_path

        candidate_buckets = list(reversed(self._list_history_dirs(run_id)))
        candidate_buckets.append(CHECKPOINT_VAL_BEST_DIRNAME)
        for bucket in candidate_buckets:
            candidate_path = self._get_checkpoint_dir(run_id, bucket)
            candidate = CheckPoint.load(candidate_path, with_states=False)
            if candidate is not None and candidate.checkpoint_id == selector:
                return candidate_path
        return None

    def clone_last_checkpoint(self, run_id: str, name: str) -> CheckPoint:
        checkpoint_name = _validate_named_checkpoint_name(name)
        source_path = self._get_last_checkpoint_dir(run_id)
        if source_path is None:
            raise ValueError(f"no checkpoint is available for run '{run_id}'")

        checkpoint = CheckPoint.load(source_path, with_states=False)
        if checkpoint is None:
            raise RuntimeError(f"failed to load latest checkpoint for run '{run_id}'")

        named_dir = self._run_dir(run_id) / CHECKPOINT_NAMED_DIRNAME
        target_path = named_dir / checkpoint_name
        if target_path.exists():
            raise ValueError(f"named checkpoint '{checkpoint_name}' already exists for run '{run_id}'")

        named_dir.mkdir(parents=True, exist_ok=True)
        temp_path = named_dir / f".{checkpoint_name}.tmp-{uuid4().hex}"
        try:
            shutil.copytree(source_path, temp_path)
            temp_path.rename(target_path)
        finally:
            if temp_path.exists():
                shutil.rmtree(temp_path)

        return checkpoint

    def _get_last_checkpoint_dir(self, run_id: str) -> Optional[str]:
        history_dirs = self._list_history_dirs(run_id)
        if len(history_dirs) == 0:
            return None
        return self._get_checkpoint_dir(run_id, history_dirs[-1])

    def for_run(self, run_id: str) -> RunCheckpoints:
        return RunCheckpoints(run_id=run_id, _store=self)


def _validate_named_checkpoint_name(name: str) -> str:
    checkpoint_name = name.strip()
    if not _NAMED_CHECKPOINT_NAME_RE.fullmatch(checkpoint_name):
        raise ValueError(
            "checkpoint name must use letters, digits, dots, underscores, or hyphens, "
            "and must start with a letter or digit"
        )
    if checkpoint_name in {"last", "best"}:
        raise ValueError(f"checkpoint name '{checkpoint_name}' is reserved")
    return checkpoint_name
