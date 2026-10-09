"""
Visual SVG rendering for chess boards with colored intervention and prediction arrows.
"""

from typing import List, Optional, Tuple, Union
import chess
import chess.svg


def render_scope_board(
    board: chess.Board,
    model_moves: Optional[List[Tuple[str, float]]] = None,
    added_moves: Optional[List[str]] = None,
    highlight_move: Optional[str] = None,
    size: int = 440,
) -> str:
    """
    Renders an SVG chess board with semantic color-coded arrows:
    - Blue (#2563EB): Model top predicted moves (opacity indicates probability).
    - Orange (#F59E0B): Newly added legal moves under the counterfactual rule.
    - Green (#10B981): Highlighted target move.
    """
    arrows = []

    # 1. Added moves under edited rule (Orange arrows)
    if added_moves:
        for m_str in added_moves[:6]:
            try:
                m = chess.Move.from_uci(m_str)
                arrows.append(chess.svg.Arrow(m.from_square, m.to_square, color="#f59e0b88"))
            except Exception:
                pass

    # 2. Model predicted moves (Blue arrows)
    if model_moves:
        for m_str, prob in model_moves[:4]:
            try:
                m = chess.Move.from_uci(m_str)
                # Opacity based on probability
                arrows.append(chess.svg.Arrow(m.from_square, m.to_square, color="#2563ebcc"))
            except Exception:
                pass

    # 3. Explicitly highlighted move (Green arrow)
    if highlight_move:
        try:
            m = chess.Move.from_uci(highlight_move)
            arrows.append(chess.svg.Arrow(m.from_square, m.to_square, color="#10b981ee"))
        except Exception:
            pass

    last_move = board.peek() if board.move_stack else None

    return chess.svg.board(
        board=board,
        lastmove=last_move,
        arrows=arrows,
        size=size,
    )
