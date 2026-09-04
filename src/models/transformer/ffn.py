import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, bias: bool = False) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class GELUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, bias: bool = True) -> None:
        super().__init__()
        self.w_in = nn.Linear(d_model, d_ff, bias=bias)
        self.w_out = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_out(F.gelu(self.w_in(x))))


def build_ffn(activation: str, d_model: int, dropout: float, bias: bool = False) -> nn.Module:
    d_ff = 4 * d_model
    if activation == "swiglu":
        d_ff = int(2 * d_model * 4 / 3)
        d_ff = 8 * ((d_ff + 7) // 8)
        return SwiGLU(d_model, d_ff, dropout, bias)
    if activation == "gelu":
        return GELUFFN(d_model, d_ff, dropout, bias)
    raise ValueError(f"Unknown activation: {activation}")
