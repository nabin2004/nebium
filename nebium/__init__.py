"""
Nebium: Self-Supervised Causal Transformer for Chess Move Prediction.

Quickstart:
    >>> import nebium
    >>> model, tokenizer = nebium.load_model("large")
    >>> prompt = "e2e4 e7e5 g1f3"
    >>> tokens = tokenizer.encode(prompt)
"""

from src.models.transformer.nebium import Nebium
from src.data.tokenizer import ChessTokenizer
from nebium.presets import (
    MODEL_PRESETS,
    PRESET_ALIASES,
    HF_REPOS,
    get_model_config,
    list_models,
    resolve_preset_key,
)
from nebium.hub import (
    build_model,
    load_model,
    load_tokenizer,
    clean_and_extract_state_dict,
)

__version__ = "0.1.0"

__all__ = [
    "Nebium",
    "ChessTokenizer",
    "load_model",
    "load_tokenizer",
    "build_model",
    "get_model_config",
    "list_models",
    "clean_and_extract_state_dict",
    "MODEL_PRESETS",
    "PRESET_ALIASES",
    "HF_REPOS",
]
