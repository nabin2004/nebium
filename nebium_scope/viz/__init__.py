"""
Visualization helpers for SVG rendering and diagnostic plotting.
"""

from nebium_scope.viz.board import render_scope_board
from nebium_scope.viz.diagrams import (
    plot_embedding_space,
    plot_activation_heatmap,
    plot_layer_dynamics,
    plot_logit_lens_trajectory,
    plot_transformer_flow_diagram,
)

__all__ = [
    "render_scope_board",
    "plot_embedding_space",
    "plot_activation_heatmap",
    "plot_layer_dynamics",
    "plot_logit_lens_trajectory",
    "plot_transformer_flow_diagram",
]
