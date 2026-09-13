from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wintermute.data.constants import DIR_EXTERNAL_MODELS, DIR_HUGGING_FACE_MODELS
from wintermute.ml.runs import ResolvedCheckpoint, RunStore
from wintermute.ml.tasks.factory import TaskWrapperFactory
from wintermute.ml.tasks.implementations.causal_lm.task import CausalLmTaskWrapper
from wintermute.tools.logging import console_log


EVALUATIONS_DIRNAME = "evaluations"


@dataclass(frozen=True)
class LmEvalTarget:
    """One saved Wintermute run or external Hugging Face model to evaluate."""

    kind: Literal["run", "huggingface"]
    name: str
    checkpoint: str = "last"

    def __post_init__(self) -> None:
        if self.kind not in {"run", "huggingface"}:
            raise ValueError(f"unsupported lm-eval target kind: {self.kind}")
        clean_name = self.name.strip()
        if not clean_name:
            raise ValueError(f"{self.kind} evaluation target name must not be empty")
        object.__setattr__(self, "name", clean_name)

    @classmethod
    def from_run(cls, run_id: str, checkpoint: str = "last") -> "LmEvalTarget":
        return cls(kind="run", name=run_id, checkpoint=checkpoint)

    @classmethod
    def from_huggingface(cls, model_name: str) -> "LmEvalTarget":
        return cls(kind="huggingface", name=model_name)


@dataclass(frozen=True)
class ResolvedLmEvalTarget:
    adapter: Any
    evaluation_root: Path
    metadata: Mapping[str, Any]
    checkpoint: ResolvedCheckpoint | None


def resolve_lm_eval_target(
    output_root: str | Path,
    target: LmEvalTarget,
    config: Mapping[str, Any],
    device: torch.device,
) -> ResolvedLmEvalTarget:
    if target.kind == "run":
        return _build_run_target(output_root, target.name, target.checkpoint, device)
    return _build_huggingface_target(output_root, target.name, config, device)


def _build_run_target(
    output_root: str | Path,
    run_id: str,
    checkpoint: str,
    device: torch.device,
) -> ResolvedLmEvalTarget:
    run = RunStore(output_root).open(run_id, load_slack_config=False)
    resolved = run.checkpoints.resolve(checkpoint, with_states=True)
    if resolved is None:
        raise ValueError(f"checkpoint '{checkpoint}' not found for run '{run_id}'")
    model_state = resolved.checkpoint.states.get("model")
    if model_state is None:
        raise ValueError(
            f"checkpoint '{resolved.checkpoint.checkpoint_id}' for run '{run_id}' has no model state"
        )

    console_log(
        "eval",
        f"resolved run-id {run_id} selector '{resolved.selector}' to "
        f"checkpoint {resolved.checkpoint.checkpoint_id} at {resolved.path.as_posix()}",
    )
    model = run.build_model()
    model.load_state_dict(model_state)
    task_wrapper = TaskWrapperFactory().build_wrapper(
        run.config.task_config,
        device,
        bool(run.config.trainer_config.use_bf16 and device.type == "cuda"),
        run.root,
    )
    if not isinstance(task_wrapper, CausalLmTaskWrapper):
        raise ValueError("lm-eval currently supports only Wintermute causal language-model runs")

    return build_live_run_target(
        run_id=run_id,
        run_path=run.path,
        model=model,
        task_wrapper=task_wrapper,
        resolved_checkpoint=resolved,
    )


def build_live_run_target(
    *,
    run_id: str,
    run_path: Path,
    model: Any,
    task_wrapper: CausalLmTaskWrapper,
    resolved_checkpoint: ResolvedCheckpoint,
) -> ResolvedLmEvalTarget:
    """Build an lm-eval target around an already-loaded training model."""

    return build_loaded_model_target(
        model_name=f"{run_id}:{resolved_checkpoint.checkpoint.checkpoint_id}",
        model=model,
        task_wrapper=task_wrapper,
        evaluation_root=run_path / EVALUATIONS_DIRNAME,
        metadata={
            "run_id": run_id,
            "checkpoint_selector": resolved_checkpoint.selector,
            "checkpoint_id": resolved_checkpoint.checkpoint.checkpoint_id,
            "checkpoint_path": resolved_checkpoint.path.resolve().as_posix(),
            "checkpoint_global_step": resolved_checkpoint.checkpoint.global_step,
            "checkpoint_units_seen": resolved_checkpoint.checkpoint.units_seen,
            "checkpoint_examples_seen": resolved_checkpoint.checkpoint.examples_seen,
            "checkpoint_created_at_utc": (
                resolved_checkpoint.checkpoint.created_at.isoformat()
            ),
        },
        checkpoint=resolved_checkpoint,
    )


def build_loaded_model_target(
    *,
    model_name: str,
    model: Any,
    task_wrapper: CausalLmTaskWrapper,
    evaluation_root: Path,
    metadata: Mapping[str, Any],
    checkpoint: ResolvedCheckpoint | None = None,
) -> ResolvedLmEvalTarget:
    """Build an lm-eval target around a model and wrapper already in memory."""

    adapter = build_loaded_model_adapter(
        model_name=model_name,
        model=model,
        task_wrapper=task_wrapper,
    )
    return ResolvedLmEvalTarget(
        adapter=adapter,
        evaluation_root=evaluation_root,
        metadata=metadata,
        checkpoint=checkpoint,
    )


def build_loaded_model_adapter(
    *,
    model_name: str,
    model: Any,
    task_wrapper: CausalLmTaskWrapper,
) -> Any:
    """Expose a model and wrapper already in memory to lm-eval."""

    from wintermute.ml.evaluation.lm_eval.adapter import WintermuteLmEvalAdapter

    model.to(task_wrapper.device)
    adapter = WintermuteLmEvalAdapter(
        model=model,
        task_wrapper=task_wrapper,
        model_name=model_name,
        max_length=min(
            int(task_wrapper.windowing_policy.window_max_len),
            int(
                getattr(
                    model,
                    "max_seq_len",
                    task_wrapper.windowing_policy.window_max_len,
                )
            ),
        ),
        progress_callback=_LmEvalProgressReporter().record_request,
    )
    return adapter


def _build_huggingface_target(
    output_root: str | Path,
    model_name: str,
    config: Mapping[str, Any],
    device: torch.device,
) -> ResolvedLmEvalTarget:
    root = Path(output_root)
    console_log("eval", f"starting lm-eval Hugging Face model: {model_name}")
    return ResolvedLmEvalTarget(
        adapter=_build_huggingface_adapter(
            model_name,
            device=device,
            batch_size=config.get("batch_size", 1),
            cache_dir=root / DIR_EXTERNAL_MODELS / DIR_HUGGING_FACE_MODELS,
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        ),
        evaluation_root=root / DIR_EXTERNAL_MODELS / "evaluations" / _artifact_name(model_name),
        metadata={"target_type": "huggingface_model", "model_name": model_name},
        checkpoint=None,
    )


def _build_huggingface_adapter(
    model_name: str,
    *,
    device: torch.device,
    batch_size: Any,
    cache_dir: Path,
    trust_remote_code: bool,
) -> Any:
    try:
        from lm_eval.models.huggingface import HFLM  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Hugging Face model evaluation requires the 'accelerate' dependency; "
            "install the Wintermute project dependencies"
        ) from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    console_log("eval", f"loading Hugging Face model: {model_name}")
    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir.as_posix(),
            trust_remote_code=trust_remote_code,
        ),
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir.as_posix(),
        trust_remote_code=trust_remote_code,
    )
    return cast(Any, HFLM)(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)


def _artifact_name(value: str) -> str:
    artifact_name = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not artifact_name:
        raise ValueError("evaluation artifact name must contain a letter or digit")
    return artifact_name


class _LmEvalProgressReporter:
    def __init__(self, *, interval_sec: float = 30.0) -> None:
        self._started_at = time.monotonic()
        self._last_reported_at = self._started_at
        self._interval_sec = interval_sec

    def record_request(self, completed_request_count: int) -> None:
        now = time.monotonic()
        if completed_request_count != 1 and now - self._last_reported_at < self._interval_sec:
            return
        elapsed_sec = max(now - self._started_at, 0.001)
        console_log(
            "eval",
            f"lm-eval progress: {completed_request_count:,} requests completed "
            f"in {elapsed_sec:.0f}s ({completed_request_count / elapsed_sec:.2f} requests/s)",
        )
        self._last_reported_at = now
