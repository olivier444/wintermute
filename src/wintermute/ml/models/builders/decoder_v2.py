from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, cast

from wintermute.ml.models.specifications import ModelSpecification
from wintermute.ml.models.builders.decoder_block import DecoderBlock, build_norm
from wintermute.ml.models.builders.rope import build_rope_cache
from wintermute.ml.models.builders.factorized_embedding import FactorizedEmbeddingHead
from wintermute.ml.models.base import BaseModel
from wintermute.tools.params import get_bool, get_float, get_optional_int, get_str


class DecoderV2(BaseModel):
    def __init__(
        self,
        spec: ModelSpecification,
    ) -> None:
        super().__init__()

        self.vocab_size = int(spec.params["vocab_size"])
        self.max_seq_len = spec.params["max_seq_len"]
        self.d_model = spec.params["d_model"]
        self.n_heads = spec.params["n_heads"]
        self.head_dim = self.d_model // self.n_heads
        self.use_checkpoint = get_bool(spec.params, "use_checkpoint", True)
        self.rope_base = get_float(spec.params, "rope_base", 10000.0)
        self.tie_embeddings = bool(spec.params["tie_embeddings"])
        self.mlp_activation = get_str(spec.params, "mlp_activation", "gelu").lower()
        self.norm = get_str(spec.params, "norm", "layer").lower()
        self.qk_norm = get_bool(spec.params, "qk_norm", False)
        self.initializer_range = get_float(spec.params, "initializer_range", 0.02)
        self.scale_residual_init = get_bool(spec.params, "scale_residual_init", True)
        n_layers = int(spec.params["n_layers"])

        emb_dim = get_optional_int(spec.params, "embedding_dim")
        if emb_dim is None:
            emb_dim = self.d_model
        self.embedding_dim = int(emb_dim)

        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")

        if self.embedding_dim > self.d_model:
            raise ValueError("embedding_dim must be <= d_model")

        self.embed_head = FactorizedEmbeddingHead(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            emb_dim=self.embedding_dim,
            tie_embeddings=self.tie_embeddings,
        )

        self.drop = nn.Dropout(spec.params["dropout"])

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=spec.params["d_model"],
                    n_heads=spec.params["n_heads"],
                    d_ff=spec.params["d_ff"],
                    dropout=spec.params["dropout"],
                    n_kv_heads=get_optional_int(spec.params, "n_kv_heads"),
                    mlp_activation=self.mlp_activation,
                    norm=self.norm,
                    qk_norm=self.qk_norm,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = build_norm(self.norm, self.d_model)

        cos, sin = build_rope_cache(
            seq_len=self.max_seq_len,
            head_dim=self.head_dim,
            device=torch.device("cpu"),
            base=self.rope_base,
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        if self.scale_residual_init:
            self._init_residual_projections(n_layers)

    def excludes_weight_decay(
        self,
        module: nn.Module,
        param_name: str,
        param: nn.Parameter,
    ) -> bool:
        output_embedding = self.embed_head.output_embedding
        return (
            output_embedding is not None
            and module is output_embedding
            and param is output_embedding.weight
        ) or super().excludes_weight_decay(module, param_name, param)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            return

        if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            nn.init.ones_(module.weight)
            bias = getattr(module, "bias", None)
            if bias is not None:
                nn.init.zeros_(bias)

    def _init_residual_projections(self, n_layers: int) -> None:
        if n_layers <= 0:
            raise ValueError("n_layers must be > 0")

        residual_std = self.initializer_range / math.sqrt(2 * n_layers)
        for block_module in self.blocks:
            block = cast(DecoderBlock, block_module)
            nn.init.normal_(block.proj.weight, mean=0.0, std=residual_std)

            down_proj = cast(Optional[nn.Linear], getattr(block.mlp, "down_proj", None))
            if down_proj is not None:
                nn.init.normal_(down_proj.weight, mean=0.0, std=residual_std)
                continue

            fc2 = cast(Optional[nn.Linear], getattr(block.mlp, "fc2", None))
            if fc2 is not None:
                nn.init.normal_(fc2.weight, mean=0.0, std=residual_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [B, T]")

        _, seqlen = input_ids.shape

        if seqlen > self.max_seq_len:
            raise ValueError(f"seqlen {seqlen} > max_seq_len {self.max_seq_len}")

        x = self.embed_head.embed(input_ids)
        x = self.drop(x)

        cos = cast(torch.Tensor, self.rope_cos)[:, :, :seqlen, :]
        sin = cast(torch.Tensor, self.rope_sin)[:, :, :seqlen, :]

        for block in self.blocks:
            if self.training and self.use_checkpoint:
                x = checkpoint(
                    block,
                    x,
                    cos,
                    sin,
                    use_reentrant=False,
                )
            else:
                x = block(x, cos, sin)

        x = self.ln_f(x)
        logits = self.embed_head.project(x)

        return logits
