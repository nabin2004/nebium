"""
NebiumScope Interactive Gradio Dashboard for Editable Chess Intelligence.
"""

from typing import Any, Dict, List, Optional, Tuple
import chess
import gradio as gr
import pandas as pd
import torch

from nebium_scope.model.adapter import NebiumAdapter
from nebium_scope.model.dummy import DummyNebiumModel
from nebium_scope.rules.variant import RULE_PRESETS, RuleVariant
from nebium_scope.rules.board import VariantBoard
from nebium_scope.rules.positions import SAMPLE_POSITIONS
from nebium_scope.interventions.steering import ActivationSteering
from nebium_scope.interventions.vectors import ConceptVectorBuilder
from nebium_scope.analysis.logit_lens import LogitLens
from nebium_scope.analysis.metrics import RuleEvaluator
from nebium_scope.viz.board import render_scope_board


def create_dashboard(
    adapter: Optional[NebiumAdapter] = None,
) -> gr.Blocks:
    """
    Builds the complete NebiumScope Gradio application.
    """
    if adapter is None:
        adapter = NebiumAdapter(DummyNebiumModel(n_layers=24, d_model=1024))

    steering = ActivationSteering(adapter.model)
    logit_lens = LogitLens(adapter)

    # Cached concept vectors per layer
    cached_vectors = {
        l: ConceptVectorBuilder.generate_mock_vector(d_model=adapter.d_model, seed=100 + l)
        for l in range(adapter.num_layers)
    }

    preset_names = [p["name"] for p in SAMPLE_POSITIONS]
    preset_fens = {p["name"]: p["fen"] for p in SAMPLE_POSITIONS}
    variant_keys = list(RULE_PRESETS.keys())

    with gr.Blocks(title="NebiumScope | Editable Chess Intelligence Workbench") as demo:
        gr.Markdown(
            """
            # 🔬 NebiumScope: Editable Chess Intelligence
            ### Interpretability, Activation Steering & Rule Variant Analysis for Nebium Transformers
            """
        )

        with gr.Tabs():
            # =================================================================
            # TAB 1: POSITION & RULE EXPLORER
            # =================================================================
            with gr.TabItem("♟️ Position & Rule Explorer"):
                with gr.Row():
                    with gr.Column(scale=4):
                        preset_dd = gr.Dropdown(
                            choices=preset_names,
                            value=preset_names[0],
                            label="Benchmark Sample Position",
                        )
                        fen_input = gr.Textbox(
                            value=preset_fens[preset_names[0]],
                            label="Position FEN or Move Sequence",
                            lines=2,
                        )
                        variant_dd = gr.Dropdown(
                            choices=variant_keys,
                            value=variant_keys[0],
                            label="Active Rule Variant",
                        )
                        analyze_btn = gr.Button("Analyze Position", variant="primary")

                        rule_desc = gr.Markdown(
                            f"**Active Rule:** {RULE_PRESETS[variant_keys[0]].title}\n\n"
                            f"*{RULE_PRESETS[variant_keys[0]].description}*"
                        )

                    with gr.Column(scale=5):
                        board_html = gr.HTML(label="Board Representation")

                with gr.Row():
                    with gr.Column():
                        normal_moves_box = gr.Textbox(label="Standard Legal Moves", lines=3)
                    with gr.Column():
                        added_moves_box = gr.Textbox(label="Newly Added Moves (Under Rule Variant)", lines=3)

                model_preds_df = gr.Dataframe(
                    label="Model Top Predictions vs Legality Status",
                    headers=["Move", "Probability", "Normal Status", "Edited Rule Status"],
                )

            # =================================================================
            # TAB 2: CAUSAL EDIT LAB (ACTIVATION STEERING)
            # =================================================================
            with gr.TabItem("🧪 Causal Edit Lab"):
                gr.Markdown(
                    """
                    **Activation Steering Intervention:**
                    Inject a contrastive rule direction $v$ at layer $L$ with strength $\\alpha$:
                    $$h_L \\leftarrow h_L + \\alpha \\cdot v$$
                    """
                )
                with gr.Row():
                    with gr.Column(scale=4):
                        steer_layer_slider = gr.Slider(
                            minimum=0,
                            maximum=max(0, adapter.num_layers - 1),
                            value=min(14, adapter.num_layers - 1),
                            step=1,
                            label="Target Intervention Layer",
                        )
                        alpha_slider = gr.Slider(
                            minimum=-3.0,
                            maximum=3.0,
                            value=1.5,
                            step=0.1,
                            label="Steering Strength (α)",
                        )
                        apply_steer_btn = gr.Button("Apply Steering & Compare", variant="primary")

                        metrics_summary = gr.Markdown("### Intervention Metrics\n*Run steering to evaluate.*")

                    with gr.Column(scale=6):
                        steered_board_html = gr.HTML(label="Steered Prediction Board")
                        deltas_table = gr.Dataframe(
                            label="Move Probability Deltas (Before vs After Steering)",
                            headers=["Move", "Category", "Baseline Prob", "Steered Prob", "Change"],
                        )

            # =================================================================
            # TAB 3: LOGIT LENS ACROSS DEPTH
            # =================================================================
            with gr.TabItem("🔍 Logit Lens Inspector"):
                gr.Markdown(
                    """
                    **Layer-Wise Logit Lens:**
                    Intermediate residual stream states projected through final normalization and LM head
                    to track where the model's move prediction forms.
                    """
                )
                run_lens_btn = gr.Button("Compute Logit Lens Across All Layers", variant="secondary")
                lens_table = gr.Dataframe(
                    label="Logit Lens Progression",
                    headers=["Layer", "Top Predicted Move", "Top Probability", "Entropy", "Top Candidates"],
                )

            # =================================================================
            # TAB 4: RULE BENCHMARK SUITE
            # =================================================================
            with gr.TabItem("📊 Rule Benchmark"):
                gr.Markdown(
                    """
                    **Batch Contrast Evaluation:**
                    Tests Rule Compliance Rate (RCR), Normal Chess Retention (NCR), and Illegal Move Rate (IMR).
                    """
                )
                run_bench_btn = gr.Button("Run Benchmark Across Sample Positions", variant="secondary")
                bench_table = gr.Dataframe(
                    label="Benchmark Results",
                    headers=["Position Name", "Has Divergence", "RCR Before", "RCR After", "NCR", "IMR"],
                )

        # ---------------------------------------------------------------------
        # Event Callbacks
        # ---------------------------------------------------------------------
        def on_preset_change(choice: str):
            fen = preset_fens.get(choice, chess.STARTING_FEN)
            return fen

        preset_dd.change(on_preset_change, inputs=[preset_dd], outputs=[fen_input])

        def on_variant_change(var_key: str):
            meta = RULE_PRESETS[var_key]
            return f"**Active Rule:** {meta.title}\n\n*{meta.description}*"

        variant_dd.change(on_variant_change, inputs=[variant_dd], outputs=[rule_desc])

        def update_position_view(fen: str, var_key: str):
            variant = RULE_PRESETS[var_key]
            vb = VariantBoard(fen, variant=variant)
            diff = vb.evaluate_move_sets()

            preds = adapter.predict_moves(fen, top_k=6)
            board_svg = render_scope_board(
                vb.board,
                model_moves=preds,
                added_moves=diff.added_moves,
                size=440,
            )

            # Prepare predictions table
            norm_set = set(diff.normal_moves)
            edit_set = set(diff.edited_moves)
            pred_rows = []
            for m, prob in preds:
                is_norm = "Legal" if m in norm_set else "Illegal"
                is_edit = "Legal (Rule)" if m in edit_set else "Illegal"
                pred_rows.append([m, f"{prob * 100:.2f}%", is_norm, is_edit])

            normal_str = ", ".join(diff.normal_moves[:20]) + ("..." if len(diff.normal_moves) > 20 else "")
            added_str = ", ".join(diff.added_moves) if diff.added_moves else "(No new moves in this position)"

            df = pd.DataFrame(pred_rows, columns=["Move", "Probability", "Normal Status", "Edited Rule Status"])
            return board_svg, normal_str, added_str, df

        analyze_btn.click(
            update_position_view,
            inputs=[fen_input, variant_dd],
            outputs=[board_html, normal_moves_box, added_moves_box, model_preds_df],
        )

        def apply_steering_view(fen: str, var_key: str, layer: int, alpha: float):
            variant = RULE_PRESETS[var_key]
            vb = VariantBoard(fen, variant=variant)
            diff = vb.evaluate_move_sets()

            # 1. Baseline predictions
            base_preds = adapter.predict_moves(fen, top_k=8)

            # 2. Steered predictions
            vec = cached_vectors.get(layer, cached_vectors[0])
            with steering.apply(layer=layer, vector=vec, alpha=alpha):
                steer_preds = adapter.predict_moves(fen, top_k=8)

            # 3. Quantitative metrics
            metrics = RuleEvaluator.evaluate(base_preds, steer_preds, diff)
            df = RuleEvaluator.deltas_to_dataframe(metrics)

            board_svg = render_scope_board(
                vb.board,
                model_moves=steer_preds,
                added_moves=diff.added_moves,
                size=440,
            )

            summary_md = f"""
            ### Quantitative Evaluation
            - **Rule Compliance Rate (RCR):** {metrics.rule_compliance_rate_before*100:.1f}% → **{metrics.rule_compliance_rate_after*100:.1f}%**
            - **Normal Chess Retention (NCR):** **{metrics.normal_chess_retention*100:.1f}%**
            - **Illegal Move Rate (IMR):** {metrics.illegal_move_rate_before*100:.1f}% → **{metrics.illegal_move_rate_after*100:.1f}%**
            """

            return board_svg, df, summary_md

        apply_steer_btn.click(
            apply_steering_view,
            inputs=[fen_input, variant_dd, steer_layer_slider, alpha_slider],
            outputs=[steered_board_html, deltas_table, metrics_summary],
        )

        def compute_logit_lens_view(fen: str):
            records = logit_lens.analyze(fen, top_k=3)
            return logit_lens.to_dataframe(records)

        run_lens_btn.click(compute_logit_lens_view, inputs=[fen_input], outputs=[lens_table])

        def run_benchmark_view(var_key: str, layer: int, alpha: float):
            variant = RULE_PRESETS[var_key]
            rows = []
            vec = cached_vectors.get(layer, cached_vectors[0])

            for p in SAMPLE_POSITIONS:
                vb = VariantBoard(p["fen"], variant=variant)
                diff = vb.evaluate_move_sets()
                base_preds = adapter.predict_moves(p["fen"], top_k=6)
                with steering.apply(layer=layer, vector=vec, alpha=alpha):
                    steer_preds = adapter.predict_moves(p["fen"], top_k=6)
                m = RuleEvaluator.evaluate(base_preds, steer_preds, diff)
                rows.append([
                    p["name"],
                    "Yes" if diff.has_rule_divergence else "No",
                    f"{m.rule_compliance_rate_before*100:.1f}%",
                    f"{m.rule_compliance_rate_after*100:.1f}%",
                    f"{m.normal_chess_retention*100:.1f}%",
                    f"{m.illegal_move_rate_after*100:.1f}%",
                ])

            return pd.DataFrame(
                rows,
                columns=["Position Name", "Has Divergence", "RCR Before", "RCR After", "NCR", "IMR"],
            )

        run_bench_btn.click(
            run_benchmark_view,
            inputs=[variant_dd, steer_layer_slider, alpha_slider],
            outputs=[bench_table],
        )

        # Initial trigger on load
        demo.load(
            update_position_view,
            inputs=[fen_input, variant_dd],
            outputs=[board_html, normal_moves_box, added_moves_box, model_preds_df],
        )

    return demo


def launch(
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = False,
) -> None:
    """
    Launches the NebiumScope dashboard.
    """
    adapter = NebiumAdapter(model=model, tokenizer=tokenizer)
    demo = create_dashboard(adapter)
    demo.launch(server_port=port, share=share, inbrowser=inbrowser)


if __name__ == "__main__":
    launch()
