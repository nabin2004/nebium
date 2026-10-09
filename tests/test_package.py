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
    tokens = m.generate_moves("", tok, max_new_moves=1)
    assert isinstance(tokens, str)


def test_utils_board_and_legal_moves():
    board = nebium.utils.get_board("e2e4 e7e5 g1f3")
    assert board.fen().startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R")
    assert nebium.utils.is_legal_move(board, "b8c6") is True
    assert nebium.utils.is_legal_move(board, "e8e1") is False

    legal_moves = nebium.utils.get_legal_moves(board)
    assert "b8c6" in legal_moves
    assert len(legal_moves) == 29


def test_utils_board_to_tensor():
    board = nebium.utils.get_board("e2e4")
    tensor = nebium.utils.board_to_tensor(board)
    assert tensor.shape == (12, 8, 8)
    assert tensor.dtype == torch.float32
    # Check that white pawn moved to e4 (rank 3, file 4 in 0-indexed coords)
    assert tensor[0, 3, 4] == 1.0


def test_utils_visualization():
    ascii_board = nebium.utils.render_ascii("e2e4")
    assert "a b c d e f g h" in ascii_board
    svg_board = nebium.utils.render_svg("e2e4")
    assert "<svg" in svg_board


def test_utils_debug_position():
    diag = nebium.utils.debug_position(
        "e2e4 e7e5",
        candidate_moves=["d2d4", "a1a8"],
        verbose=False,
    )
    assert diag["board"].turn == 1  # chess.WHITE
    assert len(diag["valid_moves"]) == 2
    assert len(diag["predictions"]) == 2
    # d2d4 is legal
    d2d4_entry = next(p for p in diag["predictions"] if p["move"] == "d2d4")
    assert d2d4_entry["is_legal"] is True
    # a1a8 is illegal
    a1a8_entry = next(p for p in diag["predictions"] if p["move"] == "a1a8")
    assert a1a8_entry["is_legal"] is False

