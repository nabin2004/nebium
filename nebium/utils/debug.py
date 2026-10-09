"""
Move debugging, model prediction inspection, and error diagnosis utilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import chess

from nebium.utils.board import (
    board_from_moves,
    board_to_features,
    board_to_tensor,
    get_material_balance,
)
from nebium.utils.visualization import render_ascii


def diagnose_move(board: chess.Board, move_uci: str) -> Dict[str, Any]:
    """
    Diagnoses why a candidate move is legal or illegal in the given board position.

    Returns:
        Dict with 'is_legal', 'reason', and 'san' if legal.
    """
    clean_uci = move_uci.strip().lower()
    try:
        move_obj = chess.Move.from_uci(clean_uci)
    except Exception as e:
        return {
            "move": clean_uci,
            "is_legal": False,
            "san": None,
            "reason": f"Malformed UCI move syntax ({e})",
        }

    if move_obj in board.legal_moves:
        return {
            "move": clean_uci,
            "is_legal": True,
            "san": board.san(move_obj),
            "reason": "Legal move",
        }

    # Pinpoint why it is illegal
    from_sq = move_obj.from_square
    to_sq = move_obj.to_square
    piece = board.piece_at(from_sq)

    if piece is None:
        reason = f"No piece at source square {chess.square_name(from_sq)}"
    elif piece.color != board.turn:
        color_str = "White" if board.turn == chess.WHITE else "Black"
        piece_color = "Black" if board.turn == chess.WHITE else "White"
        reason = f"Source square contains {piece_color} piece, but it is {color_str}'s turn"
    else:
        # Check if pseudo-legal (geometry allowed but leaves king in check)
        pseudo_legal = move_obj in board.pseudo_legal_moves
        if pseudo_legal:
            reason = "Pseudo-legal move, but leaves or keeps King in check"
        else:
            reason = f"Invalid piece movement geometry for {piece.symbol()} from {chess.square_name(from_sq)} to {chess.square_name(to_sq)}"

    return {
        "move": clean_uci,
        "is_legal": False,
        "san": None,
        "reason": reason,
    }


def debug_position(
    prompt: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    candidate_moves: Optional[List[str]] = None,
    top_k: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive diagnostics for a move sequence and model predictions.

    Args:
        prompt: Space-separated UCI move sequence (e.g. 'e2e4 e7e5 g1f3').
        model: Optional Nebium model instance for generating next move predictions.
        tokenizer: Optional ChessTokenizer instance.
        candidate_moves: Optional list of candidate UCI moves to validate manually.
        top_k: Number of model candidate moves to evaluate.
        verbose: If True, prints a formatted report to stdout.

    Returns:
        Dictionary containing board state, features, predictions, and tensor planes.
    """
    board, valid_moves, invalid_moves = board_from_moves(prompt, strict=False)
    features = board_to_features(board)
    tensor = board_to_tensor(board)

    predictions: List[Dict[str, Any]] = []

    # 1. Model inference if provided
    if model is not None and tokenizer is not None:
        try:
            preds = model.predict_next_moves(prompt, tokenizer, top_k=top_k)
            for move_str, prob in preds:
                diag = diagnose_move(board, move_str)
                diag["probability"] = prob
                predictions.append(diag)
        except Exception as e:
            if verbose:
                print(f"[Warning] Model prediction failed: {e}")

    # 2. Additional manual candidate moves if provided
    if candidate_moves is not None:
        for m in candidate_moves:
            if not any(p["move"] == m for p in predictions):
                diag = diagnose_move(board, m)
                diag["probability"] = None
                predictions.append(diag)

    # Verbose diagnostic report
    if verbose:
        ascii_grid = render_ascii(board)
        turn_str = "White to move" if board.turn == chess.WHITE else "Black to move"
        check_str = " [CHECK]" if board.is_check() else ""
        mate_str = " [CHECKMATE]" if board.is_checkmate() else ""

        print("=" * 65)
        print(" NEBIUM CHESS POSITION DIAGNOSTICS")
        print("=" * 65)
        print(f"Sequence played : {prompt or '(empty / startpos)'}")
        print(f"Valid moves     : {len(valid_moves)} moves")
        if invalid_moves:
            print(f"Invalid moves   : {invalid_moves}")
        print(f"Current Turn    : {turn_str}{check_str}{mate_str}")
        print(f"FEN             : {board.fen()}")
        print(f"Material Diff   : {features['material']['differential']:+d} (W: {features['material']['white_material']}, B: {features['material']['black_material']})")
        print(f"Total Legal     : {features['num_legal_moves']} legal moves available")
        print("\n" + ascii_grid + "\n")

        if predictions:
            print("-" * 65)
            print(f"{'MOVE':<8} | {'STATUS':<8} | {'PROBABILITY':<12} | {'DIAGNOSTIC REASON'}")
            print("-" * 65)
            for p in predictions:
                status = "LEGAL" if p["is_legal"] else "ILLEGAL"
                prob_str = f"{p['probability'] * 100:.2f}%" if p.get("probability") is not None else "N/A"
                print(f"{p['move']:<8} | {status:<8} | {prob_str:<12} | {p['reason']}")
            print("-" * 65)

    return {
        "board": board,
        "valid_moves": valid_moves,
        "invalid_moves": invalid_moves,
        "features": features,
        "predictions": predictions,
        "tensor": tensor,
    }
