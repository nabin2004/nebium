"""
Board state tracking, move validation, and tensor feature extraction utilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import chess
import torch


PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def _ensure_board(moves_or_board: Union[str, List[str], chess.Board]) -> chess.Board:
    """Helper to convert input into a chess.Board instance."""
    if isinstance(moves_or_board, chess.Board):
        return moves_or_board.copy()
    board, _, _ = board_from_moves(moves_or_board, strict=False)
    return board


def board_from_moves(
    moves: Union[str, List[str]],
    strict: bool = False,
) -> Tuple[chess.Board, List[str], List[str]]:
    """
    Replays a move sequence on a chess.Board and tracks valid vs invalid moves.

    Args:
        moves: Space-separated UCI move string (e.g. 'e2e4 e7e5 g1f3') or list of UCI moves.
        strict: If True, raises ValueError on the first illegal or malformed move.

    Returns:
        Tuple of (board, valid_uci_moves, invalid_uci_moves).
    """
    board = chess.Board()
    if isinstance(moves, str):
        move_list = [m.strip() for m in moves.strip().split() if m.strip()]
    else:
        move_list = list(moves)

    valid_moves: List[str] = []
    invalid_moves: List[str] = []

    for m in move_list:
        try:
            move_obj = chess.Move.from_uci(m)
            if move_obj in board.legal_moves:
                board.push(move_obj)
                valid_moves.append(m)
            else:
                invalid_moves.append(m)
                if strict:
                    raise ValueError(
                        f"Move '{m}' is illegal in position: {board.fen()} after moves: {' '.join(valid_moves)}"
                    )
        except Exception as e:
            invalid_moves.append(m)
            if strict:
                raise ValueError(f"Malformed UCI move '{m}': {e}") from e

    return board, valid_moves, invalid_moves


def get_board(moves: Union[str, List[str]]) -> chess.Board:
    """
    Convenience function returning the chess.Board for a move sequence.
    """
    board, _, _ = board_from_moves(moves, strict=False)
    return board


def get_legal_moves(moves_or_board: Union[str, List[str], chess.Board]) -> List[str]:
    """
    Returns list of all legal moves in UCI format for the given position.
    """
    board = _ensure_board(moves_or_board)
    return [move.uci() for move in board.legal_moves]


def is_legal_move(
    moves_or_board: Union[str, List[str], chess.Board],
    candidate_move: str,
) -> bool:
    """
    Checks whether candidate_move (e.g. 'e2e4') is legal in the specified position.
    """
    board = _ensure_board(moves_or_board)
    try:
        move_obj = chess.Move.from_uci(candidate_move.strip())
        return move_obj in board.legal_moves
    except Exception:
        return False


def get_material_balance(moves_or_board: Union[str, List[str], chess.Board]) -> Dict[str, Any]:
    """
    Calculates material counts and differential for White and Black.
    """
    board = _ensure_board(moves_or_board)
    white_score = 0
    black_score = 0
    piece_counts: Dict[str, int] = {
        "P": 0, "N": 0, "B": 0, "R": 0, "Q": 0,
        "p": 0, "n": 0, "b": 0, "r": 0, "q": 0,
    }

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        symbol = piece.symbol()
        piece_counts[symbol] = piece_counts.get(symbol, 0) + 1
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white_score += val
        else:
            black_score += val

    return {
        "white_material": white_score,
        "black_material": black_score,
        "differential": white_score - black_score,
        "piece_counts": piece_counts,
    }


def board_to_features(moves_or_board: Union[str, List[str], chess.Board]) -> Dict[str, Any]:
    """
    Extracts high-level board state features into a structured dictionary.
    """
    board = _ensure_board(moves_or_board)
    material = get_material_balance(board)

    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "fullmove_number": board.fullmove_number,
        "halfmove_clock": board.halfmove_clock,
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_game_over": board.is_game_over(),
        "num_legal_moves": board.legal_moves.count(),
        "legal_moves": [m.uci() for m in board.legal_moves],
        "has_castling_rights": {
            "white_kingside": board.has_kingside_castling_rights(chess.WHITE),
            "white_queenside": board.has_queenside_castling_rights(chess.WHITE),
            "black_kingside": board.has_kingside_castling_rights(chess.BLACK),
            "black_queenside": board.has_queenside_castling_rights(chess.BLACK),
        },
        "material": material,
    }


def board_to_tensor(
    moves_or_board: Union[str, List[str], chess.Board],
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Encodes chess board position into a standard 12-channel binary piece plane tensor (12, 8, 8).

    Channels:
        0-5: White P, N, B, R, Q, K
        6-11: Black p, n, b, r, q, k

    Shape:
        (12, 8, 8) where spatial dimensions are (rank, file), 0-indexed.

    Returns:
        torch.FloatTensor of shape (12, 8, 8).
    """
    board = _ensure_board(moves_or_board)
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            tensor[plane, rank, file] = 1.0

    if device is not None:
        tensor = tensor.to(device)
    return tensor
