"""
Lightweight mock Nebium model for local testing, CI, and rapid visualization prototyping.
"""

from typing import Any, List, Optional, Tuple
import torch
from torch import nn


class DummyBlock(nn.Module):
    """Mock transformer block simulating residual hidden state transformations."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Simulate slight transformation with residual connection
        return x + 0.05 * torch.tanh(self.linear(x))


class DummyNebiumModel(nn.Module):
    """
    Mock Nebium transformer model supporting 24 layers and 1024 hidden dims
    without heavy weight checkpoints or GPU requirements.
    """

    def __init__(
        self,
        n_layers: int = 24,
        d_model: int = 1024,
        vocab_size: int = 5000,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([DummyBlock(d_model) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_embed(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def predict_next_moves(
        self,
        prompt: str,
        tokenizer: Any = None,
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """Mock candidate predictions including common chess moves."""
        # Realistic move candidates for testing
        candidates = ["e2e4", "d2d4", "g1f3", "c2c4", "b1c3", "e7e5", "e5e4", "e7e6"]
        probs = [0.42, 0.28, 0.14, 0.08, 0.05, 0.02, 0.01, 0.00]
        return list(zip(candidates[:top_k], probs[:top_k]))
