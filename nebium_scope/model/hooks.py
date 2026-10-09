"""
Context managers and hook containers for memory-safe activation recording.
"""

from typing import Any, Dict, List, Optional
import torch
from torch import nn


class ActivationRecorder:
    """
    Context manager to record hidden state activations from transformer blocks
    without memory leaks or persistent graph references.

    Args:
        model: Nebium or DummyNebiumModel instance with .blocks attribute.
        layers: Optional list of layer indices to record. If None, records all layers.
        clone: Whether to clone recorded tensors (recommended to prevent in-place mutation).
        detach: Whether to detach tensors from the computation graph.
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[int]] = None,
        clone: bool = True,
        detach: bool = True,
    ) -> None:
        self.model = model
        self.blocks = getattr(model, "blocks", [])
        self.target_layers = layers if layers is not None else list(range(len(self.blocks)))
        self.clone = clone
        self.detach = detach
        self.activations: Dict[int, torch.Tensor] = {}
        self._handles: List[Any] = []

    def _make_hook(self, layer_idx: int):
        def hook(module, input_args, output):
            # Transformer blocks output either tensor or tuple (hidden, ...)
            hidden = output[0] if isinstance(output, tuple) else output
            if self.detach:
                hidden = hidden.detach()
            if self.clone:
                hidden = hidden.clone()
            self.activations[layer_idx] = hidden
        return hook

    def __enter__(self) -> "ActivationRecorder":
        self.activations.clear()
        self._handles.clear()
        for idx in self.target_layers:
            if 0 <= idx < len(self.blocks):
                handle = self.blocks[idx].register_forward_hook(self._make_hook(idx))
                self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_at_token(self, layer_idx: int, token_idx: int = -1) -> Optional[torch.Tensor]:
        """Returns the recorded hidden state at a specific token position (default: last token)."""
        if layer_idx in self.activations:
            return self.activations[layer_idx][:, token_idx, :]
        return None
