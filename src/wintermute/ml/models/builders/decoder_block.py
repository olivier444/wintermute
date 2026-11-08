from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wintermute.ml.models.builders.rope import apply_rope
from wintermute.ml.models.builders.mlp import build_mlp
from wintermute.ml.models.model_output import LayerKvCache


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
        x, _ = self._forward(x, cos, sin)
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: LayerKvCache | None,
        cache_capacity: int,
    ) -> tuple[torch.Tensor, LayerKvCache]:
        x, present_key_value = self._forward(
            x,
            cos,
            sin,
            past_key_value,
            cache_capacity,
        )
        assert present_key_value is not None
        return x, present_key_value

    def _forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor, # kv cache offset is already applied here - shape is 1 x 1 x T x D
        sin: torch.Tensor, # kv cache offset is already applied here - shape is 1 x 1 x T x D
        past_key_value: LayerKvCache | None = None,
        cache_capacity: int | None = None,
    ) -> tuple[torch.Tensor, LayerKvCache | None]:
        # B = batch size
        # T = current input length: full sequence without a past cache,
        #     new tokens only with a past cache
        # C = d_model
        # H = q head count
        # G = kv head count ... R = H / G (query head per kv head)
        # D = head dim ... C = H * D
        # P = previous tokens (already in kv cache) ... total seq len = P + T

        # compute:
        #    x1 = input + attention(norm1(input))
        #    output = x1 + mlp(norm2(x1))

        B, T, C = x.shape       # B x T x C

        # ---- attention ----

        h = self.ln1(x)         # B x T x C

        q = self.q_proj(h)      # B x T x (H * D)
        k = self.k_proj(h)      # B x T x (G * D)
        v = self.v_proj(h)      # B x T x (G * D)

        q = q.view(B, T, self.n_heads, self.head_dim)       # B x T x H x D
        k = k.view(B, T, self.n_kv_heads, self.head_dim)    # B x T x G x D
        v = v.view(B, T, self.n_kv_heads, self.head_dim)    # B x T x G x D

        q = q.transpose(1, 2)      # B x H x T x D
        k = k.transpose(1, 2)      # B x G x T x D
        v = v.transpose(1, 2)      # B x G x T x D

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        present_key_value = None
        if cache_capacity is not None:
            if past_key_value is None:
                key_cache = k.new_empty((B, self.n_kv_heads, cache_capacity, self.head_dim))
                value_cache = v.new_empty((B, self.n_kv_heads, cache_capacity, self.head_dim))
                past_length = 0
            else:
                key_cache = past_key_value.key
                value_cache = past_key_value.value
                past_length = past_key_value.length

            cache_end = past_length + T
            key_cache[:, :, past_length:cache_end, :].copy_(k)
            value_cache[:, :, past_length:cache_end, :].copy_(v)
            present_key_value = LayerKvCache(
                key=key_cache,
                value=value_cache,
                length=cache_end,
            )
            k = key_cache[:, :, :cache_end, :]
            v = value_cache[:, :, :cache_end, :]

        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.q_per_kv, dim=1) # B x H x (P + T) x D
            v = v.repeat_interleave(self.q_per_kv, dim=1) # B x H x (P + T) x D

        attn = F.scaled_dot_product_attention(
            q,                                      # B x H x T x D
            k,                                      # B x H x (P + T) x D
            v,                                      # B x H x (P + T) x D
            # Cached decoding uses no mask because DecoderV2 guarantees T == 1.
            is_causal=past_key_value is None,
            dropout_p=self.dropout if self.training else 0.0,
        ) # same as q: B x H x T x D

        attn = attn.transpose(1, 2).contiguous()  # B x T x H x D
        attn = attn.view(B, T, C) # B x T x (H * D) = B x T x C
        x = x + self.proj(attn)

        # ---- mlp ----

        h = self.ln2(x)
        h = self.mlp(h)         # B x T x C
        x = x + h               # B x T x C

        return x, present_key_value
