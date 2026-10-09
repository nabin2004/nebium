"""
Evaluation metrics for measuring rule compliance, retention, and side-effects.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import pandas as pd

from nebium_scope.rules.board import VariantMoveSet


@dataclass
class InterventionMetrics:
    """
    Summary evaluation metrics for an edited intelligence intervention.
    """
    rule_compliance_rate_before: float
    rule_compliance_rate_after: float
    normal_chess_retention: float
    illegal_move_rate_before: float
    illegal_move_rate_after: float
    top_divergence_moves: List[Dict[str, Any]]


class RuleEvaluator:
    """
    Evaluator calculating quantitative metrics for counterfactual rule interventions.
    """

    @staticmethod
    def evaluate(
        baseline_preds: List[Tuple[str, float]],
        steered_preds: List[Tuple[str, float]],
        move_set: VariantMoveSet,
    ) -> InterventionMetrics:
        """
        Computes RCR, NCR, and IMR across baseline and steered predictions.
        """
        base_dict = {m: p for m, p in baseline_preds}
        steer_dict = {m: p for m, p in steered_preds}
        all_moves = set(base_dict.keys()) | set(steer_dict.keys())

        normal_set = set(move_set.normal_moves)
        added_set = set(move_set.added_moves)
        edited_set = set(move_set.edited_moves)

        # 1. Rule Compliance Rate (probability mass on newly enabled edited moves)
        rcr_before = sum(p for m, p in baseline_preds if m in added_set)
        rcr_after = sum(p for m, p in steered_preds if m in added_set)

        # 2. Normal Chess Retention (probability mass preserved on original legal moves)
        norm_mass_before = sum(p for m, p in baseline_preds if m in normal_set)
        norm_mass_after = sum(p for m, p in steered_preds if m in normal_set)
        ncr = (norm_mass_after / norm_mass_before) if norm_mass_before > 1e-6 else 1.0

        # 3. Illegal Move Rate (probability on moves illegal under BOTH normal and edited rules)
        imr_before = sum(p for m, p in baseline_preds if m not in edited_set and m not in normal_set)
        imr_after = sum(p for m, p in steered_preds if m not in edited_set and m not in normal_set)

        # 4. Deltas per move
        deltas = []
        for m in all_moves:
            p_base = base_dict.get(m, 0.0)
            p_steer = steer_dict.get(m, 0.0)
            diff = p_steer - p_base
            deltas.append({
                "move": m,
                "prob_before": p_base,
                "prob_after": p_steer,
                "delta": diff,
                "is_added_rule": m in added_set,
                "is_normal_legal": m in normal_set,
            })

        deltas.sort(key=lambda x: abs(x["delta"]), reverse=True)

        return InterventionMetrics(
            rule_compliance_rate_before=rcr_before,
            rule_compliance_rate_after=rcr_after,
            normal_chess_retention=min(1.0, max(0.0, ncr)),
            illegal_move_rate_before=imr_before,
            illegal_move_rate_after=imr_after,
            top_divergence_moves=deltas,
        )

    @staticmethod
    def deltas_to_dataframe(metrics: InterventionMetrics) -> pd.DataFrame:
        """Converts move delta records to a formatted pandas DataFrame."""
        rows = []
        for d in metrics.top_divergence_moves:
            status = "Added (Rule)" if d["is_added_rule"] else ("Normal Legal" if d["is_normal_legal"] else "Illegal")
            rows.append({
                "Move": d["move"],
                "Category": status,
                "Baseline Prob": f"{d['prob_before'] * 100:.2f}%",
                "Steered Prob": f"{d['prob_after'] * 100:.2f}%",
                "Change": f"{d['delta'] * 100:+.2f}%",
            })
        return pd.DataFrame(rows)
