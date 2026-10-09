"""
Loading and deployment routines for Nebium models and tokenizers.
Seamlessly retrieves weights from Hugging Face Hub or local checkpoints.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from src.models.transformer.nebium import Nebium
from src.data.tokenizer import ChessTokenizer
from nebium.presets import (
    MODEL_PRESETS,
    PRESET_ALIASES,
    get_model_config,
    resolve_preset_key,
)


def clean_and_extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    Extracts and standardizes state_dict across PyTorch checkpoint formats,
    stripping distributed training and compiler wrappers (_orig_mod., module.).
    """
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean = {}
    for k, v in state_dict.items():
        clean_key = k.replace("_orig_mod.", "").replace("module.", "")
        clean[clean_key] = v
    return clean


def build_model(size: str = "base", **overrides) -> Nebium:
    """
    Instantiates an uninitialized Nebium model for a given size preset.

    Args:
        size: Preset name ('stub', 'base', 'small', 'medium', 'large', 'xl', etc.)
        **overrides: Optional parameter overrides (e.g., dropout=0.0, max_seq_len=256)

    Returns:
        Nebium model instance.
    """
    cfg = get_model_config(size, **overrides)
    return Nebium(**cfg)


def load_tokenizer(model_name_or_path: str = "base") -> ChessTokenizer:
    """
    Loads a ChessTokenizer instance from Hugging Face Hub or a local path.

    Args:
        model_name_or_path: Preset size ('base', 'small', 'medium', 'large'),
                            Hugging Face repo ID ('nabin2004/nebium-large'),
                            or local path to tokenizer.json / directory containing it.

    Returns:
        Loaded ChessTokenizer instance.
    """
    tokenizer = ChessTokenizer()
    p = Path(model_name_or_path)

    # 1. Local path
    if p.exists():
        if p.is_dir():
            cand = p / "tokenizer.json"
            if cand.exists():
                tokenizer.load(str(cand))
                return tokenizer
        elif p.is_file():
            tokenizer.load(str(p))
            return tokenizer

    # 2. Hugging Face Hub resolution
    repo_id = _resolve_repo_id(model_name_or_path)
    from huggingface_hub import hf_hub_download

    tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    tokenizer.load(tok_path)
    return tokenizer


def load_model(
    model_name_or_path: str = "base",
    device: Optional[Union[str, torch.device]] = None,
    return_tokenizer: bool = True,
    weights_filename: str = "model.pt",
    config_filename: str = "model_config.json",
    tokenizer_filename: str = "tokenizer.json",
    size: Optional[str] = None,
    **kwargs,
) -> Union[Nebium, Tuple[Nebium, ChessTokenizer]]:
    """
    Loads a Nebium model (and optionally tokenizer) from Hugging Face Hub or a local checkpoint.

    Args:
        model_name_or_path: Preset name ('small', 'base', 'medium', 'large'),
                            Hugging Face repository ID ('nabin2004/nebium-large'),
                            or local path to a checkpoint (.pt) or export directory.
        device: Target device (torch.device or 'cuda', 'cpu'). Defaults to CUDA if available.
        return_tokenizer: Whether to return (model, tokenizer) tuple or just model.
        weights_filename: Filename of PyTorch weights file on HF Hub (default: 'model.pt').
        config_filename: Filename of config file on HF Hub (default: 'model_config.json').
        tokenizer_filename: Filename of tokenizer on HF Hub (default: 'tokenizer.json').
        size: Optional fallback model size preset if loading from a raw state_dict without config.

    Returns:
        (model, tokenizer) tuple if return_tokenizer is True, else model.

    Example:
        >>> import nebium
        >>> model, tokenizer = nebium.load_model("large")
        >>> model = nebium.load_model("small", return_tokenizer=False)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    path_obj = Path(model_name_or_path)
    model = None
    tokenizer = None

    # Case A: Local directory or file
    if path_obj.exists():
        if path_obj.is_dir():
            ckpt_path = path_obj / weights_filename
            if not ckpt_path.exists():
                for alt in ["model.pt", "best_model.pt", "checkpoint.pt"]:
                    if (path_obj / alt).exists():
                        ckpt_path = path_obj / alt
                        break

            cfg_path = path_obj / config_filename
            if not cfg_path.exists():
                for alt in ["model_config.json", "config.json"]:
                    if (path_obj / alt).exists():
                        cfg_path = path_obj / alt
                        break

            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg.pop("_target_", None)
            elif size is not None:
                cfg = get_model_config(size)
            else:
                cfg = get_model_config("base")

            model = Nebium(**cfg)
            raw_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            model.load_state_dict(clean_and_extract_state_dict(raw_ckpt), strict=False)

            if return_tokenizer:
                tok_path = path_obj / tokenizer_filename
                if tok_path.exists():
                    tokenizer = ChessTokenizer()
                    tokenizer.load(str(tok_path))
                else:
                    tokenizer = load_tokenizer("base")

        elif path_obj.is_file():
            # Single checkpoint file
            cfg_cand = path_obj.parent / config_filename
            if cfg_cand.exists():
                with open(cfg_cand, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg.pop("_target_", None)
            elif size is not None:
                cfg = get_model_config(size)
            else:
                cfg = get_model_config("base")

            model = Nebium(**cfg)
            raw_ckpt = torch.load(str(path_obj), map_location="cpu", weights_only=False)
            model.load_state_dict(clean_and_extract_state_dict(raw_ckpt), strict=False)

            if return_tokenizer:
                tok_cand = path_obj.parent / tokenizer_filename
                if tok_cand.exists():
                    tokenizer = ChessTokenizer()
                    tokenizer.load(str(tok_cand))
                else:
                    tokenizer = load_tokenizer("base")

    # Case B: Hugging Face Hub (preset alias or explicit repo_id)
    if model is None:
        repo_id = _resolve_repo_id(model_name_or_path)
        from huggingface_hub import hf_hub_download

        # 1. Config
        try:
            cfg_file = hf_hub_download(repo_id=repo_id, filename=config_filename)
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg.pop("_target_", None)
        except Exception:
            preset_key = resolve_preset_key(model_name_or_path)
            cfg = get_model_config(preset_key or "base")

        model = Nebium(**cfg)

        # 2. Weights
        weight_file = hf_hub_download(repo_id=repo_id, filename=weights_filename)
        raw_ckpt = torch.load(weight_file, map_location="cpu", weights_only=False)
        model.load_state_dict(clean_and_extract_state_dict(raw_ckpt), strict=False)

        # 3. Tokenizer
        if return_tokenizer:
            try:
                tok_file = hf_hub_download(repo_id=repo_id, filename=tokenizer_filename)
                tokenizer = ChessTokenizer()
                tokenizer.load(tok_file)
            except Exception:
                tokenizer = load_tokenizer("base")

    model.to(device)
    model.eval()

    if return_tokenizer:
        return model, tokenizer
    return model


def _resolve_repo_id(identifier: str) -> str:
    """Resolves a preset alias or repo string to a Hugging Face repository ID."""
    preset_key = resolve_preset_key(identifier)
    if preset_key is not None:
        meta = MODEL_PRESETS[preset_key]
        if meta.get("hf_repo_id") is not None:
            return meta["hf_repo_id"]
    # If not a mapped preset, treat as raw repo_id (e.g., 'username/repo')
    return identifier
