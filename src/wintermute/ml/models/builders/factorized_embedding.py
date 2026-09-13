from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedEmbeddingHead(nn.Module):
    """
    Optional low-rank factorization of token embedding / lm head.

    If emb_dim == d_model:
        - behaves like a standard embedding + linear head
    If emb_dim < d_model:
        - input path: ids -> Embedding(V, emb_dim) -> Linear(emb_dim, d_model)
        - output path: hidden -> Linear(d_model, emb_dim) -> vocab projection

    If tie_embeddings=True, output vocab projection reuses token embedding weight.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        emb_dim: int,
        tie_embeddings: bool,
    ) -> None:
        super().__init__()

        if emb_dim <= 0:
            raise ValueError("emb_dim must be > 0")
        if emb_dim > d_model:
            raise ValueError("emb_dim must be <= d_model")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.tie_embeddings = tie_embeddings

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)

        self.in_proj = (
            nn.Identity() if emb_dim == d_model else nn.Linear(emb_dim, d_model, bias=False)
        )
        self.out_proj = (
            nn.Identity() if emb_dim == d_model else nn.Linear(d_model, emb_dim, bias=False)
        )

        if tie_embeddings:
            self.output_embedding = None            
        else:
            self.output_embedding = nn.Linear(emb_dim, vocab_size, bias=False)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)   # [B, T, emb_dim]
        x = self.in_proj(x)                   # [B, T, d_model]
        return x

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.out_proj(hidden)             # [B, T, emb_dim]

        if self.output_embedding is None:
            # logits = h @ W_vocab^T
            logits = F.linear(h, self.token_embedding.weight)
        else:
            logits = self.output_embedding(h)

        return logits