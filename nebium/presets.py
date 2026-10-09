"""
Preset configurations and scaling catalog for the Nebium model family.
"""

from typing import Any, Dict

# Canonical configurations matching configs/model/*.yaml
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "nebium_stub": {
        "name": "Nebium-Stub",
        "description": "Minimal fixture model for unit tests and rapid pipeline verification.",
        "params_approx": 8_000,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 1,
        "dropout": 0.0,
        "positional_encoding": "rope",
        "attention_type": "standard",
        "activation": "swiglu",
        "norm": "rmsnorm",
        "max_seq_len": 128,
        "vocab_size": 5000,
        "gradient_checkpointing": False,
        "hf_repo_id": None,
    },
    "nebium_base": {
        "name": "Nebium-Base",
        "description": "6M parameter base Transformer for local prototyping and rapid experimentation.",
        "params_approx": 6_000_000,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "dropout": 0.1,
        "positional_encoding": "rope",
        "attention_type": "standard",
        "activation": "swiglu",
        "norm": "rmsnorm",
        "max_seq_len": 512,
        "vocab_size": 5000,
        "gradient_checkpointing": False,
        "hf_repo_id": "nabin2004/nebium",
    },
    "nebium_117m": {
        "name": "Nebium-Small",
        "description": "117M parameter causal Transformer (GPT-2 Small scale).",
        "params_approx": 117_000_000,
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "dropout": 0.1,
        "positional_encoding": "rope",
        "attention_type": "standard",
        "activation": "swiglu",
        "norm": "rmsnorm",
        "max_seq_len": 1024,
        "vocab_size": 5000,
        "gradient_checkpointing": False,
        "hf_repo_id": "nabin2004/nebium-small",
    },
    "nebium_345m": {
        "name": "Nebium-Medium",
        "description": "345M parameter causal Transformer (GPT-2 Medium scale).",
        "params_approx": 345_000_000,
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "dropout": 0.1,
        "positional_encoding": "rope",
        "attention_type": "standard",
        "activation": "swiglu",
        "norm": "rmsnorm",
        "max_seq_len": 1024,
        "vocab_size": 5000,
        "gradient_checkpointing": False,
        "hf_repo_id": "nabin2004/nebium-medium",
    },
    "nebium_762m": {
        "name": "Nebium-Large",
        "description": "762M parameter causal Transformer (GPT-2 Large scale, flagship published checkpoint).",
        "params_approx": 762_000_000,
        "d_model": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "dropout": 0.1,
        "positional_encoding": "rope",
        "attention_type": "standard",
        "activation": "swiglu",
        "norm": "rmsnorm",
        "max_seq_len": 1024,
        "vocab_size": 5000,
        "gradient_checkpointing": True,
        "hf_repo_id": "nabin2004/nebium-large",
    },
    "nebium_1_5b": {
        "name": "Nebium-XL",
        "description": "1.5B parameter causal Transformer (GPT-2 XL scale).",
        "params_approx": 1_500_000_000,
        "d_model": 1600,
        "n_heads": 25,
        "n_layers": 48,
        "dropout": 0.1,
        "positional_encoding": "rope",
        "attention_type": "standard",
        "activation": "swiglu",
        "norm": "rmsnorm",
        "max_seq_len": 1024,
        "vocab_size": 5000,
        "gradient_checkpointing": True,
        "hf_repo_id": None,
    },
}

# Aliases mapping user-friendly string identifiers to canonical keys
PRESET_ALIASES: Dict[str, str] = {
    # Stub
    "stub": "nebium_stub",
    "nebium_stub": "nebium_stub",
    "nebium-stub": "nebium_stub",
    "fixture": "nebium_stub",
    
    # Base
    "base": "nebium_base",
    "nebium_base": "nebium_base",
    "nebium-base": "nebium_base",
    "6m": "nebium_base",
    "nabin2004/nebium": "nebium_base",
    
    # Small / 117M
    "small": "nebium_117m",
    "nebium_small": "nebium_117m",
    "nebium-small": "nebium_117m",
    "117m": "nebium_117m",
    "nebium_117m": "nebium_117m",
    "nebium-117m": "nebium_117m",
    "nabin2004/nebium-small": "nebium_117m",
    
    # Medium / 345M
    "medium": "nebium_345m",
    "nebium_medium": "nebium_345m",
    "nebium-medium": "nebium_345m",
    "345m": "nebium_345m",
    "nebium_345m": "nebium_345m",
    "nebium-345m": "nebium_345m",
    "nabin2004/nebium-medium": "nebium_345m",
    
    # Large / 762M
    "large": "nebium_762m",
    "nebium_large": "nebium_762m",
    "nebium-large": "nebium_762m",
    "762m": "nebium_762m",
    "nebium_762m": "nebium_762m",
    "nebium-762m": "nebium_762m",
    "nabin2004/nebium-large": "nebium_762m",
    
    # XL / 1.5B
    "xl": "nebium_1_5b",
    "nebium_xl": "nebium_1_5b",
    "nebium-xl": "nebium_1_5b",
    "1.5b": "nebium_1_5b",
    "1_5b": "nebium_1_5b",
    "1500m": "nebium_1_5b",
    "nebium_1_5b": "nebium_1_5b",
}

HF_REPOS: Dict[str, str] = {
    "base": "nabin2004/nebium",
    "small": "nabin2004/nebium-small",
    "medium": "nabin2004/nebium-medium",
    "large": "nabin2004/nebium-large",
}


def resolve_preset_key(identifier: str) -> str | None:
    """
    Normalizes a model identifier or alias into its canonical preset key.
    
    Returns None if identifier is not a recognized preset or alias.
    """
    cleaned = identifier.strip().lower()
    return PRESET_ALIASES.get(cleaned)


def get_model_config(size: str, **overrides) -> Dict[str, Any]:
    """
    Returns a dictionary of configuration parameters for a specified Nebium model size.
    
    Args:
        size: Preset identifier ('stub', 'base', 'small', 'medium', 'large', 'xl', etc.)
        **overrides: Optional parameter overrides (e.g., dropout=0.0, max_seq_len=256)
        
    Returns:
        Dict suitable for passing into Nebium(**config).
    """
    key = resolve_preset_key(size)
    if key is None:
        raise ValueError(
            f"Unknown model size '{size}'. Available presets: {list(MODEL_PRESETS.keys())} "
            f"or aliases: {list(PRESET_ALIASES.keys())}"
        )
    
    preset = MODEL_PRESETS[key].copy()
    # Remove metadata fields not accepted by Nebium constructor
    config = {
        "vocab_size": preset["vocab_size"],
        "d_model": preset["d_model"],
        "n_heads": preset["n_heads"],
        "n_layers": preset["n_layers"],
        "dropout": preset["dropout"],
        "positional_encoding": preset["positional_encoding"],
        "attention_type": preset["attention_type"],
        "activation": preset["activation"],
        "norm": preset["norm"],
        "max_seq_len": preset["max_seq_len"],
        "gradient_checkpointing": preset.get("gradient_checkpointing", False),
    }
    config.update(overrides)
    return config


def list_models() -> Dict[str, Dict[str, Any]]:
    """
    Returns catalog describing all available model sizes, parameter counts, and HF repos.
    """
    return {
        key: {
            "name": data["name"],
            "parameters": data["params_approx"],
            "d_model": data["d_model"],
            "n_heads": data["n_heads"],
            "n_layers": data["n_layers"],
            "max_seq_len": data["max_seq_len"],
            "hf_repo_id": data["hf_repo_id"],
            "description": data["description"],
        }
        for key, data in MODEL_PRESETS.items()
    }
