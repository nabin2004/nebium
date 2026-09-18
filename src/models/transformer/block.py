"""
Transformer decoder block for Nebium.

Implements a pre-normalization residual block consisting of self-attention
followed by a feed-forward network with residual skip connections.
"""

import torch
from torch import nn

from src.models.transformer.attention import build_attention
from src.models.transformer.ffn import build_ffn
from src.models.transformer.norm import build_norm


class TransformerBlock(nn.Module):
    """
    Standard pre-normalization Transformer block.

    Applies:
        x = x + Attention(Norm(x))
        x = x + FFN(Norm(x))

    Args:
        d_model: Dimensionality of the residual stream.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length supported.
        use_rope: Whether to apply RoPE in self-attention.
        attention_type: Attention mechanism variant ('standard').
        activation: Feed-forward activation ('swiglu' or 'gelu').
        norm: Normalization variant ('rmsnorm' or 'layernorm').
        bias: Whether linear layers contain additive bias parameters.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        use_rope: bool,
        attention_type: str,
        activation: str,
        norm: str,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = build_norm(norm, d_model)
        self.attn = build_attention(attention_type, d_model, n_heads, dropout, max_seq_len, use_rope, bias)
        self.ffn_norm = build_norm(norm, d_model)
        self.ffn = build_ffn(activation, d_model, dropout, bias)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Executes pre-norm residual transformations through attention and FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            attention_mask: Binary mask of shape (batch, seq_len).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

