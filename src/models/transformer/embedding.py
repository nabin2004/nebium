import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(max_seq_len, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(hidden.size(1), device=hidden.device)
        return hidden + self.embed(positions)
