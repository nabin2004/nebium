"""
Unit tests for the top-level 'nebium' installable package and model catalog.
"""

import pytest
import torch

import nebium
from nebium import Nebium, ChessTokenizer, build_model, get_model_config, list_models


def test_package_exports():
    assert hasattr(nebium, "__version__")
    assert hasattr(nebium, "Nebium")
    assert hasattr(nebium, "ChessTokenizer")
    assert hasattr(nebium, "load_model")
    assert hasattr(nebium, "build_model")
    assert hasattr(nebium, "list_models")


def test_list_models():
    catalog = list_models()
    assert "nebium_stub" in catalog
    assert "nebium_base" in catalog
    assert "nebium_117m" in catalog
    assert "nebium_345m" in catalog
    assert "nebium_762m" in catalog
    assert "nebium_1_5b" in catalog


def test_get_model_config_aliases():
    cfg_stub = get_model_config("stub")
    assert cfg_stub["d_model"] == 64
    assert cfg_stub["n_layers"] == 1

    cfg_small = get_model_config("small")
    assert cfg_small["d_model"] == 768
    assert cfg_small["n_layers"] == 12

    cfg_large = get_model_config("large")
    assert cfg_large["d_model"] == 1280
    assert cfg_large["n_layers"] == 36


def test_build_model_presets():
    m_stub = build_model("stub")
    assert isinstance(m_stub, Nebium)
    assert m_stub.d_model == 64

    m_from_preset = Nebium.from_preset("stub")
    assert isinstance(m_from_preset, Nebium)


def test_model_inference_helpers():
    m = build_model("stub")
    tok = ChessTokenizer()
    # Dummy mock vocab for stub testing
    # Test predict_next_moves and generate_moves handles empty/mock gracefully
    tokens = m.generate_moves("", tok, max_new_moves=1)
    assert isinstance(tokens, str)
