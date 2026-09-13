from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(d_model: int, d_ff: int, mlp_activation: str) -> nn.Module:
    act = mlp_activation.lower()

    if act == "gelu":
        return GeluMLP(d_model=d_model, d_ff=d_ff)

    if act == "swiglu":
        return SwiGLUMlp(d_model=d_model, d_ff=d_ff)

    raise ValueError("mlp_activation must be one of: 'gelu', 'swiglu'")


class GeluMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class SwiGLUMlp(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        return x

