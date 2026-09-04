import torch
from torch import nn

from src.models.transformer.attention import build_attention
from src.models.transformer.ffn import build_ffn
from src.models.transformer.norm import build_norm


class TransformerBlock(nn.Module):
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
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
