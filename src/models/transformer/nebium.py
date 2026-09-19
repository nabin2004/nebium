"""
Nebium: Causal Decoder-Only Transformer for Chess Move Prediction.

Encapsulates token embeddings, stacked TransformerBlocks with RoPE and SwiGLU,
RMSNorm pre-normalization, language modeling head, and constrained decoding routines.
"""

import torch
from torch import nn

from src.models.transformer.block import TransformerBlock
from src.models.transformer.embedding import LearnedPositionalEmbedding, TokenEmbedding
from src.models.transformer.norm import build_norm


class Nebium(nn.Module):
    """
    Nebium autoregressive causal transformer.

    Args:
        vocab_size: Number of unique tokens in the vocabulary (default: 5,000).
        d_model: Hidden dimensionality of the residual stream (default: 512).
        n_heads: Number of attention heads (default: 8).
        n_layers: Number of stacked transformer layers (default: 6).
        dropout: Dropout rate applied across attention and residual pathways.
        max_seq_len: Maximum sequence context window length (default: 512).
        positional_encoding: Positional strategy, either 'rope' (default) or 'learned'.
        attention_type: Attention mechanism variant ('standard').
        activation: Feed-forward non-linearity, 'swiglu' (default) or 'gelu'.
        norm: Normalization variant, 'rmsnorm' (default) or 'layernorm'.
        bias: Whether linear projection layers contain bias parameters (default: False).
        tie_word_embeddings: Whether to tie output projection weights with token embeddings.
    """

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
        gradient_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if positional_encoding not in {"rope", "learned"}:
            raise ValueError(f"Unknown positional_encoding: {positional_encoding}")
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = gradient_checkpointing
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

    def gradient_checkpointing_enable(self) -> None:
        """Enables activation checkpointing across transformer layers to reduce training VRAM."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disables activation checkpointing."""
        self.gradient_checkpointing = False

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Applies truncated normal initialization matching standard transformer scaling."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes next-token probability logits across the sequence.

        Args:
            input_ids: Token indices tensor of shape (batch, seq_len).
            attention_mask: Binary padding mask of shape (batch, seq_len).

        Returns:
            Unnormalized logits of shape (batch, seq_len, vocab_size).
        """
        hidden = self.token_embed(input_ids)
        if self.learned_pos is not None:
            hidden = self.learned_pos(hidden)
        hidden = self.embed_dropout(hidden)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    block, hidden, attention_mask, use_reentrant=False
                )
            else:
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
        legal_tokens_fn = None,
    ) -> torch.Tensor:
        """
        Autoregressively generates next move tokens using temperature, top-k/top-p sampling,
        or dynamic legal move logit masking.

        Args:
            input_ids: Prompt token indices of shape (batch, seq_len).
            attention_mask: Prompt attention mask of shape (batch, seq_len).
            max_new_tokens: Number of subsequent tokens to generate.
            temperature: Softmax sampling temperature (0.0 for greedy decoding).
            top_k: Number of highest probability vocabulary tokens to filter.
            top_p: Nucleus sampling cumulative probability threshold.
            legal_tokens_fn: Optional callable mapping sequence IDs to allowed legal token IDs.

        Returns:
            Concatenated token tensor of shape (batch, seq_len + max_new_tokens).
        """
        tokens = input_ids
        mask = attention_mask
        for _ in range(max_new_tokens):
            cropped = tokens[:, -self.max_seq_len :]
            cropped_mask = mask[:, -self.max_seq_len :]
            logits = self.forward(cropped, cropped_mask)
            next_token_logits = logits[:, -1, :]
            
            if legal_tokens_fn is not None:
                for i in range(tokens.size(0)):
                    allowed_tokens = legal_tokens_fn(tokens[i].tolist())
                    if allowed_tokens is not None:
                        mask_legal = torch.zeros_like(next_token_logits[i], dtype=torch.bool)
                        mask_legal[allowed_tokens] = True
                        next_token_logits[i, ~mask_legal] = -float("Inf")
            
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

    @torch.no_grad()
    def beam_search_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 1,
        beam_width: int = 3,
        legal_tokens_fn = None,
    ) -> torch.Tensor:
        """
        Executes beam search decoding with cumulative log-probability ranking.

        Args:
            input_ids: Prompt token tensor of shape (1, seq_len).
            attention_mask: Binary mask of shape (1, seq_len).
            max_new_tokens: Number of future tokens to generate.
            beam_width: Number of parallel candidate beams to maintain.
            legal_tokens_fn: Optional callable filtering candidate tokens by board legality.

        Returns:
            Best sequence token tensor of shape (1, seq_len + max_new_tokens).
        """
        batch_size = input_ids.size(0)
        assert batch_size == 1, "Beam search only supports batch size 1"
        
        beams = [(0.0, input_ids[0].tolist())]
        
        for _ in range(max_new_tokens):
            new_beams = []
            for score, seq in beams:
                seq_tensor = torch.tensor([seq], dtype=torch.long, device=input_ids.device)
                mask = torch.ones_like(seq_tensor)
                
                cropped = seq_tensor[:, -self.max_seq_len :]
                cropped_mask = mask[:, -self.max_seq_len :]
                
                logits = self.forward(cropped, cropped_mask)
                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                
                if legal_tokens_fn is not None:
                    allowed = legal_tokens_fn(seq)
                    if allowed is not None:
                        mask_legal = torch.zeros_like(log_probs, dtype=torch.bool)
                        mask_legal[allowed] = True
                        log_probs[~mask_legal] = -float("Inf")
                
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                for lp, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                    if lp != -float("Inf"):
                        new_beams.append((score + lp, seq + [idx]))
                        
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]
            
        best_seq = beams[0][1]
        return torch.tensor([best_seq], dtype=torch.long, device=input_ids.device)

