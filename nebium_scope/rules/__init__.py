"""
Rule schemas and counterfactual chess engines.
"""

from nebium_scope.rules.variant import RuleVariant, RULE_PRESETS
from nebium_scope.rules.board import VariantBoard, VariantMoveSet
from nebium_scope.rules.positions import SAMPLE_POSITIONS, generate_contrast_positions

__all__ = [
    "RuleVariant",
    "RULE_PRESETS",
    "VariantBoard",
    "VariantMoveSet",
    "SAMPLE_POSITIONS",
    "generate_contrast_positions",
]
