"""
Concept vector builder for extracting contrastive rule directions across layers.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch

from nebium_scope.model.adapter import NebiumAdapter


class ConceptVectorBuilder:
    """
    Extracts and serializes contrastive steering vectors:
        v = mean(H_edited) - mean(H_normal)
    """

    @staticmethod
    def compute_vector(
        normal_hiddens: List[torch.Tensor],
        edited_hiddens: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the contrastive direction between edited and normal hidden states.
        """
        if not normal_hiddens or not edited_hiddens:
            raise ValueError("Both normal and edited hidden state lists must be non-empty.")

        d_model = normal_hiddens[0].shape[-1]
        norm_flat = torch.cat([h.reshape(-1, d_model).float() for h in normal_hiddens], dim=0)
        edit_flat = torch.cat([h.reshape(-1, d_model).float() for h in edited_hiddens], dim=0)

        mean_normal = norm_flat.mean(dim=0)
        mean_edited = edit_flat.mean(dim=0)

        diff = mean_edited - mean_normal
        # Normalize to unit length
        norm = torch.norm(diff, p=2)
        if norm > 1e-8:
            diff = diff / norm
        return diff

    @classmethod
    def extract_from_prompts(
        cls,
        adapter: NebiumAdapter,
        normal_prompts: List[str],
        edited_prompts: List[str],
        layers: Optional[List[int]] = None,
        token_pos: int = -1,
    ) -> Dict[int, torch.Tensor]:
        """
        Iterates over contrastive prompts and computes per-layer concept vectors.
        """
        target_layers = layers or list(range(adapter.num_layers))
        layer_vectors: Dict[int, torch.Tensor] = {}

        # Collect normal hidden states
        normal_states: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}
        for p in normal_prompts:
            hiddens = adapter.get_hidden_states(p, layers=target_layers)
            for l, h in hiddens.items():
                normal_states[l].append(h[:, token_pos, :])

        # Collect edited hidden states
        edited_states: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}
        for p in edited_prompts:
            hiddens = adapter.get_hidden_states(p, layers=target_layers)
            for l, h in hiddens.items():
                edited_states[l].append(h[:, token_pos, :])

        for l in target_layers:
            if normal_states[l] and edited_states[l]:
                layer_vectors[l] = cls.compute_vector(normal_states[l], edited_states[l])

        return layer_vectors

    @staticmethod
    def generate_mock_vector(d_model: int = 1024, seed: int = 42) -> torch.Tensor:
        """Generates a reproducible synthetic concept vector for local testing."""
        rng = torch.Generator().manual_seed(seed)
        vec = torch.randn(d_model, generator=rng)
        return vec / torch.norm(vec, p=2)

    @staticmethod
    def save_vectors(vectors: Dict[int, torch.Tensor], path: Union[str, Path]) -> None:
        """Saves layer vectors dictionary to disk."""
        torch.save(vectors, str(path))

    @staticmethod
    def load_vectors(path: Union[str, Path]) -> Dict[int, torch.Tensor]:
        """Loads layer vectors dictionary from disk."""
        return torch.load(str(path), map_location="cpu", weights_only=False)
