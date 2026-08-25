import torch
from torch import nn

from src.models.transformer.rope import RotaryEmbedding, apply_rotary_pos_emb


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        use_rope: bool,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len) if use_rope else None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))
        if self.rope is not None:
            cos, sin = self.rope(query.size(-2))
            query = apply_rotary_pos_emb(query, cos, sin)
            key = apply_rotary_pos_emb(key, cos, sin)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(_attention_mask(attention_mask), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(query)
        weights = weights.masked_fill(attention_mask[:, None, :, None] == 0, 0.0)
        weights = self.attn_dropout(weights)
        attended = torch.matmul(weights, value)
        return self.resid_dropout(self.out_proj(self._merge_heads(attended)))


def _attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = attention_mask.size(-1)
    causal = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device),
        diagonal=1,
    )
    padding = attention_mask[:, None, None, :] == 0
    return causal | padding


def build_attention(
    attention_type: str,
    d_model: int,
    n_heads: int,
    dropout: float,
    max_seq_len: int,
    use_rope: bool,
) -> nn.Module:
    if attention_type == "standard":
        return MultiHeadSelfAttention(d_model, n_heads, dropout, max_seq_len, use_rope)
    raise ValueError(f"Unknown attention_type: {attention_type}")
