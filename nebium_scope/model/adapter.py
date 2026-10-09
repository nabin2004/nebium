"""
ModelAdapter interface connecting Nebium transformer models and tokenizers.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from nebium_scope.model.hooks import ActivationRecorder
from nebium_scope.model.dummy import DummyNebiumModel


class NebiumAdapter:
    """
    Adapter encapsulating model inference, token encoding, and activation recording.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model is not None else DummyNebiumModel()
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    @property
    def num_layers(self) -> int:
        """Returns total number of transformer layers."""
        return len(getattr(self.model, "blocks", []))

    @property
    def d_model(self) -> int:
        """Returns residual stream dimension."""
        return getattr(self.model, "d_model", 1024)

    def prepare_input(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a space-separated UCI prompt into input_ids and attention_mask tensors.
        """
        prompt_str = prompt.strip()
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            encoded = self.tokenizer.encode(prompt_str) if prompt_str else []
            bos = getattr(self.tokenizer, "bos_id", None)
            tokens = ([bos] if bos is not None else []) + encoded
            if not tokens:
                tokens = [0]
        else:
            # Fallback mock encoding for dummy models
            tokens = [1] + [sum(ord(ch) for ch in move) % 500 for move in prompt_str.split()]
            if not tokens:
                tokens = [1]


        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def get_hidden_states(
        self,
        prompt: str,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Runs a forward pass and extracts hidden states for requested layers.

        Returns:
            Dict mapping layer index to tensor of shape (batch, seq_len, d_model).
        """
        input_ids, attention_mask = self.prepare_input(prompt)
        recorder = ActivationRecorder(self.model, layers=layers)
        with torch.no_grad(), recorder:
            _ = self.model(input_ids, attention_mask)
        return recorder.activations

    def get_logits(self, prompt: str) -> torch.Tensor:
        """Runs a forward pass and returns next-token logits of shape (1, vocab_size)."""
        input_ids, attention_mask = self.prepare_input(prompt)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
        return logits[:, -1, :]

    def predict_moves(
        self,
        prompt: str,
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """
        Generates top-k move predictions with probabilities.
        """
        if hasattr(self.model, "predict_next_moves"):
            return self.model.predict_next_moves(
                prompt=prompt,
                tokenizer=self.tokenizer,
                top_k=top_k,
                temperature=temperature,
            )
        # General fallback using model logits
        logits = self.get_logits(prompt)
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)[0]
        top_probs, top_ids = torch.topk(probs, min(top_k, probs.size(0)))

        results = []
        for idx, p in zip(top_ids, top_probs):
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                token = self.tokenizer.decode([idx.item()]).strip()
            else:
                token = f"tok_{idx.item()}"
            results.append((token, p.item()))
        return results
