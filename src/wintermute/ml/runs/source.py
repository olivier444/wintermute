from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from wintermute.tools.files import json_save


SOURCE_DIRNAME = "source"
SOURCE_CAPTURE_VERSION = 1


@dataclass(frozen=True)
class SourceCapture:
    relative_path: str
    commit: str | None


def capture_run_source(
    run_dir: str | Path,
    *,
    operation: str,
    checkpoint_selector: str | None = None,
    repository_hint: str | Path | None = None,
) -> SourceCapture:
    source_dir = Path(run_dir) / SOURCE_DIRNAME
    source_dir.mkdir(parents=True, exist_ok=True)
    capture_id = _next_capture_id(source_dir)
    hint = Path(repository_hint) if repository_hint is not None else Path(__file__).resolve().parent

    try:
        commit, branch, dirty = _read_git_state(hint)
        error = None
    except RuntimeError as exc:
        commit, branch, dirty = None, None, None
        error = str(exc)

    file_name = f"{capture_id}.json"
    json_save(
        {
            "version": SOURCE_CAPTURE_VERSION,
            "capture_id": capture_id,
            "operation": operation,
            "checkpoint_selector": checkpoint_selector,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "available": commit is not None,
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "reproducible": commit is not None and dirty is False,
            "error": error,
        },
        source_dir.as_posix(),
        file_name,
    )
    return SourceCapture(
        relative_path=f"{SOURCE_DIRNAME}/{file_name}",
        commit=commit,
    )


def _next_capture_id(source_dir: Path) -> str:
    existing_ids = [int(path.stem) for path in source_dir.glob("*.json") if path.stem.isdigit()]
    return f"{max(existing_ids, default=0) + 1:04d}"


def _read_git_state(repository_hint: Path) -> tuple[str, str | None, bool]:
    repository_root_value = _git(repository_hint, "rev-parse", "--show-toplevel")
    if not repository_root_value:
        raise RuntimeError("git rev-parse --show-toplevel returned no value")
    repository_root = Path(repository_root_value)

    commit = _git(repository_root, "rev-parse", "HEAD")
    if not commit:
        raise RuntimeError("git rev-parse HEAD returned no value")
    branch = _git(
        repository_root,
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
        allow_failure=True,
    ) or None
    dirty = bool(
        _git(
            repository_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
        )
    )
    return commit, branch, dirty


def _git(repository_path: Path, *args: str, allow_failure: bool = False) -> str:
    try:
        process = subprocess.run(
            ["git", "-C", repository_path.as_posix(), *args],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"unable to run git: {exc}") from exc

    if process.returncode != 0:
        if allow_failure:
            return ""
        detail = process.stderr.strip() or f"exit code {process.returncode}"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return process.stdout.strip()
