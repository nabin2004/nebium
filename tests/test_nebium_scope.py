"""
Unit test suite for the NebiumScope editable intelligence and interpretability kit.
"""

import pytest
import torch

import nebium_scope as ns


def test_rule_presets_and_variant_board():
    assert "pawn_backward_one" in ns.RULE_PRESETS
    assert "knight_diagonal" in ns.RULE_PRESETS

    # Position with lone White pawn on e5 and Black king on e1
    fen = "8/8/8/4P3/8/4K3/8/4k3 w - - 0 1"
    vb = ns.VariantBoard(fen, variant="pawn_backward_one")
    diff = vb.evaluate_move_sets()

    assert diff.has_rule_divergence is True
    # e5e4 is the newly legal backward pawn move
    assert "e5e4" in diff.added_moves
    # e5e6 is the normal forward pawn move
    assert "e5e6" in diff.normal_moves
    assert "e5e6" in diff.edited_moves


def test_knight_diagonal_variant():
    # Position with White Knight on d5
    fen = "8/8/8/3N4/8/4K3/8/4k3 w - - 0 1"
    vb = ns.VariantBoard(fen, variant="knight_diagonal")
    diff = vb.evaluate_move_sets()

    assert diff.has_rule_divergence is True
    # d5 can jump 2 squares diagonally: b7, f7, b3, f3
    for alfil_move in ["d5b7", "d5f7", "d5b3", "d5f3"]:
        assert alfil_move in diff.added_moves


def test_model_adapter_and_dummy_model():
    dummy = ns.DummyNebiumModel(n_layers=6, d_model=128)
    adapter = ns.NebiumAdapter(model=dummy)

    assert adapter.num_layers == 6
    assert adapter.d_model == 128

    hiddens = adapter.get_hidden_states("e2e4 e7e5", layers=[0, 3, 5])
    assert 0 in hiddens
    assert 3 in hiddens
    assert 5 in hiddens
    assert hiddens[0].shape[-1] == 128

    preds = adapter.predict_moves("e2e4", top_k=3)
    assert len(preds) == 3


def test_activation_steering_intervention():
    dummy = ns.DummyNebiumModel(n_layers=4, d_model=64)
    adapter = ns.NebiumAdapter(model=dummy)
    steering = ns.ActivationSteering(dummy)

    vec = torch.randn(64)
    with steering.apply(layer=2, vector=vec, alpha=1.5):
        h_steered = adapter.get_hidden_states("e2e4", layers=[2])
        assert h_steered[2].shape[-1] == 64

    # Verify hooks are removed after context exit
    assert len(steering._handles) == 0


def test_concept_vector_builder():
    h_normal = [torch.randn(1, 10, 64) for _ in range(3)]
    h_edited = [torch.randn(1, 10, 64) for _ in range(3)]

    vec = ns.ConceptVectorBuilder.compute_vector(h_normal, h_edited)
    assert vec.shape == (64,)
    # Should be normalized to ~1.0
    assert pytest.approx(torch.norm(vec).item(), abs=1e-4) == 1.0


def test_logit_lens():
    dummy = ns.DummyNebiumModel(n_layers=4, d_model=64)
    adapter = ns.NebiumAdapter(model=dummy)
    lens = ns.LogitLens(adapter)

    records = lens.analyze("e2e4", top_k=2)
    assert len(records) == 4
    df = lens.to_dataframe(records)
    assert len(df) == 4
    assert "Top Predicted Move" in df.columns


def test_rule_evaluator_metrics():
    baseline_preds = [("e5e6", 0.70), ("e2e4", 0.20), ("a1a8", 0.10)]
    steered_preds = [("e5e4", 0.60), ("e5e6", 0.30), ("e2e4", 0.10)]

    move_set = ns.VariantMoveSet(
        normal_moves=["e5e6", "e2e4"],
        edited_moves=["e5e6", "e2e4", "e5e4"],
        added_moves=["e5e4"],
        removed_moves=[],
    )

    metrics = ns.RuleEvaluator.evaluate(baseline_preds, steered_preds, move_set)
    assert metrics.rule_compliance_rate_before == 0.0
    assert metrics.rule_compliance_rate_after == 0.60
    assert metrics.normal_chess_retention > 0.0
    assert len(metrics.top_divergence_moves) > 0


def test_gradio_dashboard_smoke():
    dummy = ns.DummyNebiumModel(n_layers=2, d_model=64)
    adapter = ns.NebiumAdapter(model=dummy)
    demo = ns.create_dashboard(adapter)
    assert demo is not None
