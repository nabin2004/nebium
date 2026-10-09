"""
Position generators for rule variant benchmarking and contrast-pair creation.
"""

from typing import Any, Dict, List
import chess

from nebium_scope.rules.board import VariantBoard
from nebium_scope.rules.variant import RuleVariant, RULE_PRESETS

# Benchmark positions curated for testing rule divergence
SAMPLE_POSITIONS: List[Dict[str, str]] = [
    {
        "name": "Advanced White Pawn (e5)",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "description": "Italian opening with white pawn on e4; White to move.",
    },
    {
        "name": "Pawn Chain (e4/d4 vs e5/d5)",
        "fen": "rnbqkbnr/pp3ppp/2p5/3pp3/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4",
        "description": "Pawn tensions in the center; backward pawn move on d4 or e4 opens new retreats.",
    },
    {
        "name": "King & Pawn Endgame (e5 passed pawn)",
        "fen": "8/8/8/4P3/8/4K3/8/4k3 w - - 0 1",
        "description": "Endgame position with lone white pawn on e5 with empty square on e4 behind it.",
    },
    {
        "name": "Knight Jump Outpost (d5)",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/3Np3/4P3/8/PPPP1PPP/R1BQKBNR w KQkq - 2 4",
        "description": "White Knight on d5; tests knight diagonal leaps to b7/f7/b3/f3.",
    },
    {
        "name": "Standard Starting Position",
        "fen": chess.STARTING_FEN,
        "description": "Initial board position before any moves are played.",
    },
]


def generate_contrast_positions(
    variant: RuleVariant = RULE_PRESETS["pawn_backward_one"],
    max_positions: int = 10,
) -> List[Dict[str, Any]]:
    """
    Generates a collection of benchmark positions containing active rule divergences.
    """
    results = []
    for item in SAMPLE_POSITIONS:
        vb = VariantBoard(item["fen"], variant=variant)
        diff = vb.evaluate_move_sets()
        results.append({
            "name": item["name"],
            "fen": item["fen"],
            "description": item["description"],
            "has_divergence": diff.has_rule_divergence,
            "normal_moves": diff.normal_moves,
            "edited_moves": diff.edited_moves,
            "added_moves": diff.added_moves,
            "removed_moves": diff.removed_moves,
        })
        if len(results) >= max_positions:
            break
    return results
