import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: [..., head_dim]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y = torch.empty_like(x)
    y[..., ::2] = -x2
    y[..., 1::2] = x1
    return y


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")

    half_dim = head_dim // 2

    freq_seq = torch.arange(half_dim, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))   # [half_dim]

    t = torch.arange(seq_len, device=device, dtype=dtype)  # [T]
    freqs = torch.outer(t, inv_freq)                       # [T, half_dim]

    # on duplique pour retrouver head_dim
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)   # [T, head_dim]

    cos = emb.cos()[None, None, :, :]   # [1, 1, T, head_dim]
    sin = emb.sin()[None, None, :, :]   # [1, 1, T, head_dim]
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # x:   [B, H, T, D]
    # cos: [1, 1, T, D]
    # sin: [1, 1, T, D]
    return (x * cos) + (rotate_half(x) * sin)