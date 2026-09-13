import importlib
import inspect
import os
import pkgutil
import time
from typing import List, Any, Iterable, Iterator, Optional, Sequence, Type, TypeVar
from datetime import datetime, timezone
from pathlib import Path
import json
from dataclasses import dataclass

from wintermute.tools.logging import console_log
from wintermute.tools.params import parse_optional_positive_int as _parse_optional_positive_int

T = TypeVar("T")


def compute_max_threads() -> int:
    return max(1, (os.cpu_count() or 0) - 2)


def compute_max_memory_gb() -> int:
    return 48


def batch_sequence(lst: Sequence[Any], k: int) -> Iterator[Sequence[Any]]:
    if k <= 0:
        raise ValueError(f"Invalid size: {k}")
        
    for i in range(0, len(lst), k):
        yield lst[i:i+k]


def batch_iter(iterator: Iterable[Any], k: int) -> Iterator[List[Any]]:
    if k <= 0:
        raise ValueError(f"Invalid size: {k}")
    
    batch: List[Any] = []
    for element in iterator:
        batch.append(element)
        if len(batch) == k:
            yield batch
            batch = []
    
    if len(batch) > 0:
        yield batch
            

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_unix_ms() -> int:
    return dt_to_unix_ms(datetime.now(timezone.utc))


def dt_to_unix_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("No timezone found")
    
    return int(dt.timestamp() * 1000)


def unix_ms_to_dt(ms: int, tz: timezone=timezone.utc) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=tz)


def json_default(o: Any) -> Any:
    # Best-effort JSON encoding for configs / dataclasses / pydantic
    if hasattr(o, "model_dump"):  # pydantic v2
        return o.model_dump()
    if hasattr(o, "dict"):  # pydantic v1
        return o.dict()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def resolve_class_from_module(
    module_name: str,
    class_name: str,
    expected_base_class: Optional[Type[Any]] = None,
) -> Type[Any]:
    module = importlib.import_module(module_name)
    candidate = getattr(module, class_name, None)
    if not inspect.isclass(candidate):
        raise KeyError(f"Unknown class '{class_name}' in module '{module_name}'.")

    if expected_base_class is not None and not issubclass(candidate, expected_base_class):
        raise TypeError(
            f"Class '{class_name}' in module '{module_name}' must inherit from '{expected_base_class.__name__}'."
        )

    return candidate


def resolve_class_from_package(
    package_name: str,
    class_name: str,
    expected_base_class: Optional[Type[Any]] = None,
) -> Type[Any]:
    package = importlib.import_module(package_name)
    module_names = [package_name]

    if not hasattr(package, "__path__"):
        return resolve_class_from_module(
            module_name=package_name,
            class_name=class_name,
            expected_base_class=expected_base_class,
        )

    for mod_info in pkgutil.walk_packages(package.__path__, f"{package_name}."):
        module_names.append(mod_info.name)

    for module_name in module_names:
        try:
            candidate = resolve_class_from_module(
                module_name=module_name,
                class_name=class_name,
                expected_base_class=expected_base_class
            )
        except KeyError:
            continue

        return candidate

    raise KeyError(f"Unknown class '{class_name}' in package '{package_name}'.")


def or_raise(opt: Optional[T], msg: str = "Unexpected None") -> T:
    if opt is None:
        raise ValueError(msg)
    return opt


def parse_optional_positive_int(value: Any, *, field_name: str) -> Optional[int]:
    return _parse_optional_positive_int(value, field_name=field_name)

@dataclass
class FileControllerResult:
    Active: bool
    Params: Optional[dict[str, Any]]


class FileController:
    def __init__(self, control_dir: Path) -> None:
        self.control_dir = control_dir
        self._last_check_times: dict[str, float] = {}
        self._last_check_values: dict[str, bool] = {}        
        self._last_check_data: dict[str, Optional[dict[str, Any]]] = {}           

    def request_path(self, name: str) -> Path:
        return self.control_dir / name
    
    def has_request(self, request: str, *, clear: bool = True, log_changes: bool = True, check_interval_sec: float = 10) -> bool:    
        return self.get_request(request, clear=clear, log_changes=log_changes, check_interval_sec=check_interval_sec).Active
    
    def get_request(self, request: str, *, clear: bool = True, log_changes: bool = True, check_interval_sec: float = 10) -> FileControllerResult:
        now = time.monotonic()
        last_checked = self._last_check_times.get(request, 0.0)
        previous_state = self._last_check_values.get(request, False)   
        previous_data = self._last_check_data.get(request, None)     

        if now - last_checked < check_interval_sec:
            return FileControllerResult(previous_state, previous_data)

        file_path = self.request_path(request)
        if not file_path.exists():
            return self._process_new_state(request, previous_state, False, None, log_changes, now, store=True)

        new_data = None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            new_data = json.loads(content) if content.strip() else {}
        except Exception as exc:
            console_log(
                "file controller",
                f"failed to load JSON request file '{file_path}': {exc}; using empty dict",
            )
            new_data = {}

        if clear:
            try:
                file_path.unlink()
            except Exception:
                pass

        return self._process_new_state(request, previous_state, True, new_data, log_changes, now, store=not clear)

    def check_pause(self, log_changes: bool = True, check_interval_sec = 10) -> bool:   
        result = False 
        while self.has_request(".pause", clear=False, log_changes=log_changes, check_interval_sec=check_interval_sec * 0.95):
            result = True
            time.sleep(check_interval_sec) 

        return result   
    
    def _process_new_state(self, request: str, previous_state: bool, new_state: bool, new_data: Optional[dict[str, Any]], log_changes: bool, now: float, store: bool) -> FileControllerResult:
        self._last_check_times[request] = now   
             
        if store:
            self._last_check_data[request] = new_data
            self._last_check_values[request] = new_state

        if log_changes and new_state != previous_state:
            state_str = "detected" if new_state else "cleared"
            console_log("file controller", f"request '{request}' {state_str}")

        return FileControllerResult(new_state, new_data)
