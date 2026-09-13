from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wintermute.ml.models.builders.rope import apply_rope
from wintermute.ml.models.builders.mlp import build_mlp


def build_norm(norm: str, d_model: int) -> nn.Module:
    if norm not in {"layer", "rms"}:
        raise ValueError("norm must be one of: 'layer', 'rms'")

    return nn.LayerNorm(d_model) if norm == "layer" else nn.RMSNorm(d_model)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        n_kv_heads: int | None = None,
        mlp_activation: str = "gelu",
        norm: str = "layer",
        qk_norm: bool = False,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.qk_norm = qk_norm

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")

        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires head_dim to be even")

        self.q_per_kv = n_heads // self.n_kv_heads

        self.ln1 = build_norm(norm, d_model)

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        self.ln2 = build_norm(norm, d_model)

        self.mlp = build_mlp(
            d_model=d_model,
            d_ff=d_ff,
            mlp_activation=mlp_activation,
        )

        self.dropout = dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # ---- attention ----

        h = self.ln1(x)

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos[:, :, :T, :], sin[:, :, :T, :])
        k = apply_rope(k, cos[:, :, :T, :], sin[:, :, :T, :])

        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.q_per_kv, dim=1)
            v = v.repeat_interleave(self.q_per_kv, dim=1)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(attn)

        # ---- mlp ----

        h = self.ln2(x)
        h = self.mlp(h)
        x = x + h

        return x
