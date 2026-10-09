"""
NebiumScope: Interactive Workbench for Editable Chess Intelligence & Circuit Analysis.

Quickstart:
    >>> import nebium_scope as ns
    >>> # 1. Test counterfactual rule divergence on a position:
    >>> board = ns.VariantBoard("8/8/8/4P3/8/4K3/8/4k3 w - - 0 1", variant="pawn_backward_one")
    >>> diff = board.evaluate_move_sets()
    >>> print("Added moves:", diff.added_moves)  # ['e5e4']
    >>>
    >>> # 2. Launch the Gradio developer dashboard:
    >>> ns.launch()
"""

from nebium_scope.rules.variant import RuleVariant, RULE_PRESETS
from nebium_scope.rules.board import VariantBoard, VariantMoveSet
from nebium_scope.rules.positions import SAMPLE_POSITIONS, generate_contrast_positions

from nebium_scope.model.hooks import ActivationRecorder
from nebium_scope.model.dummy import DummyNebiumModel, DummyBlock
from nebium_scope.model.adapter import NebiumAdapter

from nebium_scope.interventions.steering import ActivationSteering
from nebium_scope.interventions.vectors import ConceptVectorBuilder

from nebium_scope.analysis.logit_lens import LogitLens, LayerLensRecord
from nebium_scope.analysis.metrics import RuleEvaluator, InterventionMetrics

from nebium_scope.viz.board import render_scope_board
from nebium_scope.app import create_dashboard, launch

__version__ = "0.1.0"

__all__ = [
    "RuleVariant",
    "RULE_PRESETS",
    "VariantBoard",
    "VariantMoveSet",
    "SAMPLE_POSITIONS",
    "generate_contrast_positions",
    "ActivationRecorder",
    "DummyNebiumModel",
    "DummyBlock",
    "NebiumAdapter",
    "ActivationSteering",
    "ConceptVectorBuilder",
    "LogitLens",
    "LayerLensRecord",
    "RuleEvaluator",
    "InterventionMetrics",
    "render_scope_board",
    "create_dashboard",
    "launch",
]
