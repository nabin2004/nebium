"""
Embedding layers for Nebium.

Provides token embedding and learned absolute positional embeddings (used in baseline configurations).
"""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """
    Standard vocabulary token embedding layer.

    Args:
        vocab_size: Total vocabulary size (e.g. 5,000 for Nebium BPE).
        d_model: Hidden feature dimension.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Maps discrete token indices to continuous embedding vectors.

        Args:
            input_ids: Tensor of shape (batch, seq_len) with integer token IDs.

        Returns:
            Tensor of shape (batch, seq_len, d_model).
        """
        return self.embed(input_ids)


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned absolute positional embeddings (GPT-2 baseline style).

    Args:
        max_seq_len: Maximum sequence length supported.
        d_model: Hidden feature dimension.
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(max_seq_len, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Adds learned positional embeddings to input token representations.

        Args:
            hidden: Input representations of shape (batch, seq_len, d_model).

        Returns:
            Position-augmented tensor of shape (batch, seq_len, d_model).
        """
        positions = torch.arange(hidden.size(1), device=hidden.device)
        return hidden + self.embed(positions)

