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
        bias: bool = False,
        tie_word_embeddings: bool = False,
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
                    bias=bias,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = build_norm(norm, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.token_embed.embedding.weight
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
    def generate(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        max_new_tokens: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        tokens = input_ids
        mask = attention_mask
        for _ in range(max_new_tokens):
            cropped = tokens[:, -self.max_seq_len :]
            cropped_mask = mask[:, -self.max_seq_len :]
            logits = self.forward(cropped, cropped_mask)
            next_token_logits = logits[:, -1, :]
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")
                
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float("Inf")
            
            if temperature == 0.0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            tokens = torch.cat([tokens, next_token], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)
        return tokens
