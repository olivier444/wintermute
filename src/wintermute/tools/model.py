from __future__ import annotations

from contextlib import contextmanager
import gc
import random
from typing import TYPE_CHECKING, Iterator, Iterable, List, Any, cast

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer

if TYPE_CHECKING:
    from wintermute.ml.training.logger import Logger


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@contextmanager
def evaluating(model: nn.Module) -> Iterator[None]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()


@contextmanager
def preserving_torch_rng_state() -> Iterator[None]:
    cpu_rng_state = torch.get_rng_state().clone()
    cuda_rng_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else None
    )
    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(list(cuda_rng_states))


@contextmanager
def preserving_rng_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with preserving_torch_rng_state():
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def log_cuda_memory(logger: Logger, step: int, device: int = 0, prefix: str = "gpu"):
    if not torch.cuda.is_available():
        return
    dev = torch.device(f"cuda:{device}")
    alloc = torch.cuda.memory_allocated(dev) / 1024**2
    reserved = torch.cuda.memory_reserved(dev) / 1024**2
    max_alloc = torch.cuda.max_memory_allocated(dev) / 1024**2
    max_reserved = torch.cuda.max_memory_reserved(dev) / 1024**2

    logger.log_scalar(f"{prefix}/mem_allocated_mb", alloc, step)
    logger.log_scalar(f"{prefix}/mem_reserved_mb", reserved, step)
    logger.log_scalar(f"{prefix}/mem_max_allocated_mb", max_alloc, step)
    logger.log_scalar(f"{prefix}/mem_max_reserved_mb", max_reserved, step)

    torch.cuda.reset_peak_memory_stats()


def unload_from_gpu(model: nn.Module) -> nn.Module:
    model = model.to("cpu")
    clear_cuda_memory()
    return model


def move_optimizer_state(optimizer: Optimizer, device: torch.device | str) -> None:
    target_device = torch.device(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=target_device)


def clear_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()


def extract_logits_from_model_output(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict) and "logits" in output:
        return cast(torch.Tensor, output["logits"])

    if isinstance(output, tuple) and output:
        return cast(torch.Tensor, output[0])

    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits

    raise TypeError(f"Unsupported model output type: {type(output)!r}")
        

def iter_batches(source: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for example in source:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
