"""
Layer-wise Logit Lens analysis for tracking decision formation across depth.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import pandas as pd

from nebium_scope.model.adapter import NebiumAdapter


@dataclass
class LayerLensRecord:
    layer: int
    top_move: str
    top_prob: float
    entropy: float
    top_3_summary: str


class LogitLens:
    """
    Projects intermediate residual stream hidden states through the final
    layer normalization and language modeling head to observe where move decisions form.
    """

    def __init__(self, adapter: NebiumAdapter) -> None:
        self.adapter = adapter
        self.model = adapter.model

    def analyze(
        self,
        prompt: str,
        token_pos: int = -1,
        top_k: int = 3,
    ) -> List[LayerLensRecord]:
        """
        Runs logit lens projection across all layers for the given prompt.
        """
        hiddens = self.adapter.get_hidden_states(prompt)
        final_norm = getattr(self.model, "final_norm", None)
        lm_head = getattr(self.model, "lm_head", None)

        results: List[LayerLensRecord] = []
        num_layers = self.adapter.num_layers

        for layer_idx in range(num_layers):
            if layer_idx not in hiddens:
                continue

            h = hiddens[layer_idx][:, token_pos, :]  # (1, d_model)

            # Project through final norm & lm head if available
            with torch.no_grad():
                if final_norm is not None:
                    h_norm = final_norm(h)
                else:
                    h_norm = h

                if lm_head is not None:
                    logits = lm_head(h_norm)
                else:
                    # Mock projection if lm_head not present
                    logits = h_norm[:, :self.adapter.model.vocab_size]

                probs = torch.softmax(logits, dim=-1)[0]
                entropy = -(probs * torch.log(probs + 1e-12)).sum().item()

                top_probs, top_ids = torch.topk(probs, min(top_k, probs.size(0)))

            candidates: List[Tuple[str, float]] = []
            for idx, p in zip(top_ids, top_probs):
                if self.adapter.tokenizer is not None and hasattr(self.adapter.tokenizer, "decode"):
                    tok = self.adapter.tokenizer.decode([idx.item()]).strip()
                else:
                    tok = f"m_{idx.item()}"
                candidates.append((tok, p.item()))

            top_move = candidates[0][0] if candidates else "none"
            top_prob = candidates[0][1] if candidates else 0.0
            summary = ", ".join(f"{m} ({p*100:.1f}%)" for m, p in candidates)

            results.append(
                LayerLensRecord(
                    layer=layer_idx,
                    top_move=top_move,
                    top_prob=top_prob,
                    entropy=entropy,
                    top_3_summary=summary,
                )
            )

        return results

    def to_dataframe(self, records: List[LayerLensRecord]) -> pd.DataFrame:
        """Converts logit lens records into a pandas DataFrame."""
        return pd.DataFrame([
            {
                "Layer": r.layer,
                "Top Predicted Move": r.top_move,
                "Top Probability": f"{r.top_prob * 100:.2f}%",
                "Entropy": f"{r.entropy:.3f}",
                "Top Candidates": r.top_3_summary,
            }
            for r in records
        ])
