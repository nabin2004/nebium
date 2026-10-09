"""
VariantBoard engine for computing legal moves under custom counterfactual rules.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple, Union
import chess

from nebium_scope.rules.variant import RuleVariant, RULE_PRESETS


@dataclass
class VariantMoveSet:
    """
    Comparison between normal chess legal moves and edited rule legal moves.
    """
    normal_moves: List[str]
    edited_moves: List[str]
    added_moves: List[str]
    removed_moves: List[str]

    @property
    def has_rule_divergence(self) -> bool:
        """True if the rule edit creates new or removes existing legal moves."""
        return bool(self.added_moves or self.removed_moves)


class VariantBoard:
    """
    Chess board wrapper providing legal move generation under standard
    and counterfactual chess rule variants.
    """

    def __init__(
        self,
        fen_or_moves: Optional[Union[str, chess.Board]] = None,
        variant: Optional[RuleVariant] = None,
    ) -> None:
        if isinstance(fen_or_moves, chess.Board):
            self.board = fen_or_moves.copy()
        elif isinstance(fen_or_moves, str) and "/" in fen_or_moves:
            self.board = chess.Board(fen_or_moves)
        elif isinstance(fen_or_moves, str) and fen_or_moves.strip():
            # Treat as UCI move sequence
            self.board = chess.Board()
            for m in fen_or_moves.strip().split():
                try:
                    self.board.push_uci(m)
                except Exception:
                    break
        else:
            self.board = chess.Board()

        if variant is not None:
            self.set_variant(variant)
        else:
            self.variant = RULE_PRESETS["pawn_backward_one"]

    def set_variant(self, variant: Union[str, RuleVariant]) -> None:
        """Sets the active rule variant by preset name or RuleVariant instance."""
        if isinstance(variant, str):
            if variant not in RULE_PRESETS:
                raise ValueError(f"Unknown variant '{variant}'. Available: {list(RULE_PRESETS.keys())}")
            self.variant = RULE_PRESETS[variant]
        else:
            self.variant = variant

    def get_normal_legal_moves(self) -> List[str]:
        """Returns standard legal moves in UCI format."""
        return [m.uci() for m in self.board.legal_moves]

    def get_edited_legal_moves(self) -> List[str]:
        """
        Computes all legal moves under the active RuleVariant.
        """
        normal_moves_set = set(self.get_normal_legal_moves())
        added_moves: Set[str] = set()
        removed_moves: Set[str] = set()

        turn = self.board.turn

        # 1. Pawn Backward Step
        if self.variant.allow_backward:
            for sq in self.board.pieces(chess.PAWN, turn):
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                backward_rank = rank - 1 if turn == chess.WHITE else rank + 1
                if 0 <= backward_rank <= 7:
                    target_sq = chess.square(file, backward_rank)
                    if self.board.piece_at(target_sq) is None:
                        # Test if playing this move leaves king in check
                        move_obj = chess.Move(sq, target_sq)
                        if self._is_safe_king_move(move_obj):
                            added_moves.add(move_obj.uci())

        # 2. Pawn Capture Forward
        if self.variant.capture_forward:
            for sq in self.board.pieces(chess.PAWN, turn):
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                forward_rank = rank + 1 if turn == chess.WHITE else rank - 1
                if 0 <= forward_rank <= 7:
                    target_sq = chess.square(file, forward_rank)
                    target_piece = self.board.piece_at(target_sq)
                    if target_piece is not None and target_piece.color != turn:
                        move_obj = chess.Move(sq, target_sq)
                        if self._is_safe_king_move(move_obj):
                            added_moves.add(move_obj.uci())

        # 3. Custom piece leap offsets (e.g. Knight diagonal leaps)
        if self.variant.custom_offsets:
            piece_type = chess.PIECE_SYMBOLS.index(self.variant.piece.lower())
            for sq in self.board.pieces(piece_type, turn):
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                for df, dr in self.variant.custom_offsets:
                    tgt_f = file + df
                    tgt_r = rank + dr
                    if 0 <= tgt_f <= 7 and 0 <= tgt_r <= 7:
                        tgt_sq = chess.square(tgt_f, tgt_r)
                        tgt_piece = self.board.piece_at(tgt_sq)
                        if tgt_piece is None or tgt_piece.color != turn:
                            move_obj = chess.Move(sq, tgt_sq)
                            if self._is_safe_king_move(move_obj):
                                added_moves.add(move_obj.uci())

        edited_set = (normal_moves_set - removed_moves) | added_moves
        return sorted(list(edited_set))

    def evaluate_move_sets(self) -> VariantMoveSet:
        """
        Returns full comparison between normal and edited rule move sets.
        """
        normal = sorted(self.get_normal_legal_moves())
        edited = sorted(self.get_edited_legal_moves())
        normal_set = set(normal)
        edited_set = set(edited)

        added = sorted(list(edited_set - normal_set))
        removed = sorted(list(normal_set - edited_set))

        return VariantMoveSet(
            normal_moves=normal,
            edited_moves=edited,
            added_moves=added,
            removed_moves=removed,
        )

    def _is_safe_king_move(self, move: chess.Move) -> bool:
        """Verifies move does not leave own king in check using speculative board state."""
        b = self.board.copy()
        # Temporarily place/move piece
        piece = b.remove_piece_at(move.from_square)
        if piece is None:
            return False
        b.set_piece_at(move.to_square, piece)
        # Check if king of moving side is attacked
        king_sq = b.king(self.board.turn)
        if king_sq is None:
            return False
        return not b.is_attacked_by(not self.board.turn, king_sq)
