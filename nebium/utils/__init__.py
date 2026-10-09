"""
Nebium Chess Utilities Submodule.

Provides board state tracking, legal move validation, 12-plane tensor extraction,
ASCII and SVG visualization, and move debugging diagnostics.
"""

from nebium.utils.board import (
    board_from_moves,
    get_board,
    get_legal_moves,
    is_legal_move,
    get_material_balance,
    board_to_features,
    board_to_tensor,
)
from nebium.utils.visualization import (
    render_ascii,
    render_svg,
    display_board,
)
from nebium.utils.debug import (
    diagnose_move,
    debug_position,
)

__all__ = [
    "board_from_moves",
    "get_board",
    "get_legal_moves",
    "is_legal_move",
    "get_material_balance",
    "board_to_features",
    "board_to_tensor",
    "render_ascii",
    "render_svg",
    "display_board",
    "diagnose_move",
    "debug_position",
]
