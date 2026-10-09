"""
Visualization and rendering utilities for chess board states.
Supports rich SVG display in Jupyter/Kaggle and clean Unicode/ASCII in terminals.
"""

from typing import Any, List, Optional, Union
import chess
import chess.svg

from nebium.utils.board import _ensure_board

UNICODE_PIECE_MAP = {
    "R": "♜", "N": "♞", "B": "♝", "Q": "♛", "K": "♚", "P": "♟",
    "r": "♖", "n": "♘", "b": "♗", "q": "♕", "k": "♔", "p": "♙",
}


def render_ascii(
    moves_or_board: Union[str, List[str], chess.Board],
    unicode_pieces: bool = False,
) -> str:
    """
    Renders an 8x8 chess board as a clean text grid with ranks and files.

    Args:
        moves_or_board: UCI move string, list of moves, or chess.Board.
        unicode_pieces: If True, uses Unicode chess glyphs. If False (default), uses letters (P, N, etc.).

    Returns:
        Formatted multi-line board string.
    """
    board = _ensure_board(moves_or_board)
    lines = []
    lines.append("  +-----------------+")
    for rank in range(7, -1, -1):
        row_str = f"{rank + 1} |"
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                char = "."
            elif unicode_pieces:
                char = UNICODE_PIECE_MAP.get(piece.symbol(), piece.symbol())
            else:
                char = piece.symbol()
            row_str += f" {char}"
        row_str += " |"
        lines.append(row_str)
    lines.append("  +-----------------+")
    lines.append("    a b c d e f g h")
    return "\n".join(lines)


def render_svg(
    moves_or_board: Union[str, List[str], chess.Board],
    last_move: Optional[Union[str, chess.Move]] = None,
    arrows: Optional[List[Any]] = None,
    size: int = 400,
) -> str:
    """
    Generates an SVG representation of the chess board.

    Args:
        moves_or_board: UCI string, list of moves, or chess.Board.
        last_move: Optional UCI move string or chess.Move to highlight.
        arrows: Optional list of chess.svg.Arrow tuples.
        size: Pixel width/height of the generated SVG image.

    Returns:
        SVG XML markup string.
    """
    board = _ensure_board(moves_or_board)
    last_move_obj = None
    if last_move is not None:
        if isinstance(last_move, str):
            try:
                last_move_obj = chess.Move.from_uci(last_move)
            except Exception:
                last_move_obj = None
        else:
            last_move_obj = last_move
    elif board.move_stack:
        last_move_obj = board.peek()

    return chess.svg.board(
        board=board,
        lastmove=last_move_obj,
        arrows=arrows or [],
        size=size,
    )


def display_board(
    moves_or_board: Union[str, List[str], chess.Board],
    size: int = 400,
) -> None:
    """
    Displays the chess board in the active environment:
    - If in a Jupyter / Kaggle / Colab notebook, renders interactive SVG.
    - Otherwise, prints the ASCII board to stdout.
    """
    try:
        from IPython.display import SVG, display  # type: ignore

        svg_content = render_svg(moves_or_board, size=size)
        display(SVG(svg_content))
    except (ImportError, Exception):
        try:
            print(render_ascii(moves_or_board, unicode_pieces=True))
        except UnicodeEncodeError:
            print(render_ascii(moves_or_board, unicode_pieces=False))

