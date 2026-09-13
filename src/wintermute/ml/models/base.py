from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for locally trainable Wintermute models."""

    def build_optimizer_param_groups(self) -> List[Dict[str, Any]]:
        params_by_id: Dict[int, Dict[str, Any]] = {}

        for _, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue

                apply_weight_decay = not self.excludes_weight_decay(module, param_name, param)

                param_id = id(param)
                if param_id not in params_by_id:
                    params_by_id[param_id] = {
                        "param": param,
                        "apply_weight_decay": apply_weight_decay,
                    }
                    continue

                if not apply_weight_decay:
                    params_by_id[param_id]["apply_weight_decay"] = False

        if not params_by_id:
            raise ValueError("Model could not build AdamW param groups: no trainable parameters found.")

        decay_params = [
            param_info["param"]
            for param_info in params_by_id.values()
            if param_info["apply_weight_decay"]
        ]
        no_decay_params = [
            param_info["param"]
            for param_info in params_by_id.values()
            if not param_info["apply_weight_decay"]
        ]
        return [
            {"params": decay_params, "apply_weight_decay": True},
            {"params": no_decay_params, "apply_weight_decay": False},
        ]

    def excludes_weight_decay(
        self,
        module: nn.Module,
        param_name: str,
        param: nn.Parameter,
    ) -> bool:
        return (
            param_name == "bias"
            or param.ndim < 2
            or isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.Embedding))
        )


def count_optimizer_param_groups(param_groups: List[Dict[str, Any]]) -> Tuple[int, int]:
    decay_param_count = 0
    no_decay_param_count = 0

    for param_group in param_groups:
        param_count = len(param_group["params"])
        if param_group.get("apply_weight_decay", True):
            decay_param_count += param_count
        else:
            no_decay_param_count += param_count

    return decay_param_count, no_decay_param_count
