"""
Feed-forward network (FFN) architectures for Nebium transformer blocks.

Implements SwiGLU (Shazeer, 2020) with parameter dimension scaling, alongside
a standard GELU baseline FFN for comparative ablation studies.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) feed-forward network.

    Computes:
        Output = (SiLU(x * W_gate) * (x * W_up)) * W_down

    Args:
        d_model: Dimensionality of the model residual stream.
        d_ff: Hidden projection dimension (typically 8/3 * d_model rounded to multiples of 8).
        dropout: Dropout probability applied to the final projection.
        bias: Whether linear layers include additive bias terms (default: False).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float, bias: bool = False) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying gated non-linear projection and down-projection.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of shape (batch, seq_len, d_model).
        """
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class GELUFFN(nn.Module):
    """
    Standard two-layer feed-forward network with GELU activation (GPT-2 baseline).

    Computes:
        Output = GELU(x * W_in) * W_out

    Args:
        d_model: Dimensionality of the model residual stream.
        d_ff: Hidden projection dimension (typically 4 * d_model).
        dropout: Dropout probability applied to the final projection.
        bias: Whether linear layers include additive bias terms (default: True).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float, bias: bool = True) -> None:
        super().__init__()
        self.w_in = nn.Linear(d_model, d_ff, bias=bias)
        self.w_out = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying GELU non-linear expansion and contraction.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of shape (batch, seq_len, d_model).
        """
        return self.dropout(self.w_out(F.gelu(self.w_in(x))))


def build_ffn(activation: str, d_model: int, dropout: float, bias: bool = False) -> nn.Module:
    """
    Factory function to construct the requested feed-forward network variant.

    Args:
        activation: Activation type, either 'swiglu' or 'gelu'.
        d_model: Model hidden dimensionality.
        dropout: Dropout rate.
        bias: Whether to use linear layer biases.

    Returns:
        Configured nn.Module instance with parameter parity scaling.

    Raises:
        ValueError: If `activation` is not recognized.
    """
    d_ff = 4 * d_model
    if activation == "swiglu":
        d_ff = int(2 * d_model * 4 / 3)
        d_ff = 8 * ((d_ff + 7) // 8)
        return SwiGLU(d_model, d_ff, dropout, bias)
    if activation == "gelu":
        return GELUFFN(d_model, d_ff, dropout, bias)
    raise ValueError(f"Unknown activation: {activation}")

