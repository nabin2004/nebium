"""
Rotary Position Embedding (RoPE) implementation for Nebium.

Implements rotary embeddings following Su et al. (2024), applying complex
rotations in 2D coordinate subspaces to encode relative token distances directly
into query and key representations.
"""

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half of the hidden dimensions by 90 degrees for RoPE computation.

    Args:
        x: Input tensor of shape (batch, heads, seq_len, head_dim).

    Returns:
        Tensor with the second half negated and swapped with the first half.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Precomputed Rotary Position Embedding cache for causal multi-head attention.

    Computes inverse frequency sinusoidal bases across feature dimensions and
    caches cosine and sine rotation matrices up to `max_seq_len`.

    Args:
        head_dim: Dimensionality of each attention head (must be even).
        max_seq_len: Maximum sequence length supported by the cache.
        base: Frequency scaling base (default: 10000.0).
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves cached cosine and sine tables truncated to the input sequence length.

        Args:
            seq_len: Number of tokens in the current sequence.

        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, head_dim).
        """
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary position embeddings to query or key tensors.

    Args:
        x: Query or key tensor of shape (batch, num_heads, seq_len, head_dim).
        cos: Cosine tensor of shape (seq_len, head_dim).
        sin: Sine tensor of shape (seq_len, head_dim).

    Returns:
        Rotated tensor matching the shape and dtype of x.
    """
    cos = cos.to(dtype=x.dtype, device=x.device)[None, None, :, :]
    sin = sin.to(dtype=x.dtype, device=x.device)[None, None, :, :]
    return x * cos + rotate_half(x) * sin

