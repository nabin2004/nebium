"""
Activation steering interventions for causal rule editing in transformer models.
"""

from typing import Any, List, Optional, Tuple, Union
import torch
from torch import nn


class ActivationSteering:
    """
    Applies reversible activation steering interventions during the model forward pass.

    Equation:
        h_layer[pos] <- h_layer[pos] + alpha * vector
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.blocks = getattr(model, "blocks", [])
        self._handles: List[Any] = []

    def _make_steering_hook(
        self,
        vector: torch.Tensor,
        alpha: float,
        token_pos: int = -1,
    ):
        def hook(module, input_args, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            steered = hidden.clone()

            # Ensure vector matches device and precision
            vec = vector.to(steered.device, dtype=steered.dtype)
            if vec.ndim == 1:
                vec = vec.unsqueeze(0)  # (1, d_model)

            # Apply steering to target token position
            steered[:, token_pos, :] += alpha * vec

            if is_tuple:
                return (steered,) + output[1:]
            return steered

        return hook

    def apply(
        self,
        layer: int,
        vector: torch.Tensor,
        alpha: float = 1.0,
        token_pos: int = -1,
    ) -> "ActivationSteering":
        """
        Attaches steering hook to target layer. Call remove() or use as context manager.
        """
        self.remove()
        if 0 <= layer < len(self.blocks):
            handle = self.blocks[layer].register_forward_hook(
                self._make_steering_hook(vector, alpha, token_pos)
            )
            self._handles.append(handle)
        return self

    def remove(self) -> None:
        """Removes all active steering hooks, reverting model to pristine baseline state."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "ActivationSteering":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()
