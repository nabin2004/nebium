"""
Rule variant definitions and configuration schemas for editable chess intelligence.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class RuleVariant:
    """
    Specification for an alternative chess rule variant.

    Attributes:
        name: Unique identifier string (e.g. 'pawn_backward_one').
        title: Human-readable display title.
        piece: Piece symbol modified ('P', 'N', 'B', 'R', 'Q', 'K').
        description: Scientific explanation of the rule modification.
        allow_backward: Whether pawns can move backward one step.
        capture_forward: Whether pawns capture forward rather than diagonally.
        custom_offsets: List of (file_offset, rank_offset) extra movement offsets.
    """
    name: str
    title: str
    piece: str
    description: str
    allow_backward: bool = False
    capture_forward: bool = False
    custom_offsets: List[tuple[int, int]] = field(default_factory=list)


RULE_PRESETS: Dict[str, RuleVariant] = {
    "pawn_backward_one": RuleVariant(
        name="pawn_backward_one",
        title="Pawns Can Move Backward 1 Square",
        piece="P",
        description="Pawns retain all standard rules, plus can step backward 1 square to an unblocked square.",
        allow_backward=True,
    ),
    "pawn_capture_forward": RuleVariant(
        name="pawn_capture_forward",
        title="Pawns Capture Straight Forward",
        piece="P",
        description="Pawns capture forward directly into an occupied enemy square rather than diagonally.",
        capture_forward=True,
    ),
    "knight_diagonal": RuleVariant(
        name="knight_diagonal",
        title="Knights Move Diagonally (Alfil Leaps)",
        piece="N",
        description="Knights can leap 2 squares diagonally (+-2, +-2) in addition to their L-shaped moves.",
        custom_offsets=[(2, 2), (2, -2), (-2, 2), (-2, -2)],
    ),
    "super_pawn": RuleVariant(
        name="super_pawn",
        title="Super Pawn (Forward & Backward Step + Forward Capture)",
        piece="P",
        description="Pawns can step backward and capture straight forward.",
        allow_backward=True,
        capture_forward=True,
    ),
}
