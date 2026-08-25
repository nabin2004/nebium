import torch
from torch import nn

from src.models.transformer.block import TransformerBlock
from src.models.transformer.embedding import LearnedPositionalEmbedding, TokenEmbedding
from src.models.transformer.norm import build_norm


class Nebium(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_seq_len: int,
        positional_encoding: str = "rope",
        attention_type: str = "standard",
        activation: str = "swiglu",
        norm: str = "rmsnorm",
        **kwargs,
    ) -> None:
        super().__init__()
        if positional_encoding not in {"rope", "learned"}:
            raise ValueError(f"Unknown positional_encoding: {positional_encoding}")
        self.max_seq_len = max_seq_len
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.learned_pos = (
            LearnedPositionalEmbedding(max_seq_len, d_model) if positional_encoding == "learned" else None
        )
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    use_rope=positional_encoding == "rope",
                    attention_type=attention_type,
                    activation=activation,
                    norm=norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = build_norm(norm, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embed(input_ids)
        if self.learned_pos is not None:
            hidden = self.learned_pos(hidden)
        hidden = self.embed_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return self.lm_head(self.final_norm(hidden))

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int = 1) -> torch.Tensor:
        tokens = input_ids
        mask = attention_mask
        for _ in range(max_new_tokens):
            cropped = tokens[:, -self.max_seq_len :]
            cropped_mask = mask[:, -self.max_seq_len :]
            logits = self.forward(cropped, cropped_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)
        return tokens
