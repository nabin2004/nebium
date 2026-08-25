import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


def build_norm(name: str, dim: int) -> nn.Module:
    if name == "rmsnorm":
        return RMSNorm(dim)
    if name == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown norm: {name}")
