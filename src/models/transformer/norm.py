"""
Normalization layer implementations for Nebium.

Provides Root Mean Square Layer Normalization (RMSNorm) following Zhang & Sennrich (2019)
for improved computational efficiency and training stability over standard LayerNorm.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes inputs by their root mean square without subtracting mean statistics:
        RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight

    Args:
        dim: Dimensionality of the normalized feature axis.
        eps: Small epsilon constant for numerical stability (default: 1e-6).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMSNorm in float32 precision and casts back to input dtype.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor matching the shape and dtype of x.
        """
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


def build_norm(name: str, dim: int) -> nn.Module:
    """
    Constructs the specified normalization layer.

    Args:
        name: Normalization variant ('rmsnorm' or 'layernorm').
        dim: Feature dimension.

    Returns:
        Configured normalization nn.Module.

    Raises:
        ValueError: If `name` is unrecognized.
    """
    if name == "rmsnorm":
        return RMSNorm(dim)
    if name == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown norm: {name}")

