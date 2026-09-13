from __future__ import annotations

from pathlib import Path


class AbstractStore:
    def __init__(
            self, 
            root_dir: str | Path
        ):
        
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)


    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])


    def get_object_dir(self, object_name: str, ensure_exist: bool=False, silent: bool = False) -> str:
        path = self.root / object_name
        if ensure_exist:
            path.mkdir(parents=True, exist_ok=True)
        else:
            if not path.exists() and not silent:
                raise ValueError(f"Path {path} does not exist")
        return path.as_posix()
