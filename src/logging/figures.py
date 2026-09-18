"""
Publication-grade figure generation for Nebium training and evaluation dynamics.

Generates high-resolution, vector-ready visualizations adhering to IEEE/Nature
academic publishing guidelines: clear typography, discrete color palettes,
subtle gridlines, and complete metric annotations.
"""

import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# Set academic visual style defaults
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.frameon": True,
    "legend.edgecolor": "#cccccc",
})


def plot_training_dynamics(
    history: list[dict[str, Any]],
    tier_name: str = "Nebium",
) -> plt.Figure:
    """
    Plots dual-panel training and validation dynamics:
      Left panel: Cross-Entropy Loss (Train vs. Validation with minimum highlighted).
      Right panel: Validation Perplexity.
    """
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_ppls = [h.get("val_ppl", math.exp(min(h["val_loss"], 20.0))) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # --- Panel 1: Loss ---
    ax1.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2.0, marker="o", markersize=4)
    ax1.plot(epochs, val_losses, label="Validation Loss", color="#d62728", linewidth=2.0, marker="s", markersize=4)

    # Annotate minimum validation loss
    if val_losses:
        min_val_idx = int(np.argmin(val_losses))
        min_epoch = epochs[min_val_idx]
        min_loss = val_losses[min_val_idx]
        ax1.scatter([min_epoch], [min_loss], color="#d62728", s=100, zorder=5, edgecolor="black", linewidth=1.2)
        ax1.annotate(
            f"Min Val: {min_loss:.4f}\n(Epoch {min_epoch})",
            xy=(min_epoch, min_loss),
            xytext=(min_epoch + 0.3, min_loss + 0.08 * (max(val_losses) - min_loss + 1e-4)),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
            fontsize=9,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff9e6", ec="#d4b106", lw=0.8),
        )

    ax1.set_title(f"{tier_name} — Cross-Entropy Loss Progression", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Optimization Epoch", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss (nats)", fontsize=10)
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=9)

    # --- Panel 2: Perplexity ---
    ax2.plot(epochs, val_ppls, label="Validation Perplexity", color="#2ca02c", linewidth=2.0, marker="^", markersize=4)
    ax2.set_title(f"{tier_name} — Perplexity Trajectory", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("Optimization Epoch", fontsize=10)
    ax2.set_ylabel("Perplexity ($\\exp(L_{val})$)", fontsize=10)
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=9)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
    except Exception:
        pass
    return fig


def plot_accuracy_and_legality(
    history: list[dict[str, Any]],
    tier_name: str = "Nebium",
) -> plt.Figure:
    """
    Plots token accuracy (Top-1 / Top-5) alongside board-validated legal move rate.
    """
    epochs = [h["epoch"] for h in history]
    accs = [h.get("val_accuracy", float("nan")) * 100 for h in history]
    top5_accs = [h.get("val_top5_accuracy", float("nan")) * 100 for h in history]
    legal_rates = [h.get("legal_move_rate", float("nan")) * 100 for h in history]

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)

    # Token accuracy on primary axis
    l1 = ax1.plot(epochs, accs, label="Top-1 Accuracy", color="#1f77b4", linewidth=2.0, marker="o", markersize=5)
    l2 = ax1.plot(epochs, top5_accs, label="Top-5 Accuracy", color="#aec7e8", linewidth=1.8, linestyle="--", marker="s", markersize=4)
    ax1.set_xlabel("Optimization Epoch", fontsize=10)
    ax1.set_ylabel("Next-Token Accuracy (%)", fontsize=10, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim([0, 100])
    ax1.grid(True)

    # Legal move rate on secondary axis
    ax2 = ax1.twinx()
    l3 = ax2.plot(epochs, legal_rates, label="Legal Move Rate (Board)", color="#ff7f0e", linewidth=2.2, marker="D", markersize=5)
    ax2.set_ylabel("Legal Move Rate (%)", fontsize=10, color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2.set_ylim([0, 105])

    # Combined legend
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", fontsize=9)

    ax1.set_title(f"{tier_name} — Token Accuracy & Legal Move Convergence", fontsize=11, fontweight="bold", pad=8)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
    except Exception:
        pass
    return fig


def plot_scaling_law_alignment(
    empirical_params: int,
    empirical_loss: float,
    current_tier: str = "Small (117M)",
    trained_tokens: int = 12_000_000,
) -> plt.Figure:
    """
    Plots theoretical Chinchilla scaling law curves across parameter counts
    alongside the 3 Nebium models and current empirical result.
    """
    def chinchilla_loss(N: np.ndarray, D: float) -> np.ndarray:
        return 1.69 + 406.4 / (N ** 0.34) + 410.7 / (D ** 0.28)

    # Parameter range: 10M to 1.5B
    N_range = np.logspace(7, 9.2, 200)

    # Chinchilla-optimal compute (D* = 20N)
    L_optimal = 1.69 + 406.4 / (N_range ** 0.34) + 410.7 / ((20 * N_range) ** 0.28)
    # Fixed-budget compute (e.g. current token horizon)
    L_fixed_tokens = chinchilla_loss(N_range, max(trained_tokens, 1_000_000))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.plot(N_range / 1e6, L_optimal, label="Chinchilla Optimal ($D^* = 20N$)", color="#2ca02c", linewidth=2.0, linestyle="-")
    ax.plot(N_range / 1e6, L_fixed_tokens, label=f"Chinchilla at Fixed Tokens ($D = {trained_tokens/1e6:.1f}$M)", color="#1f77b4", linewidth=1.8, linestyle="--")

    # Mark the three Nebium family tiers
    tiers = [
        ("Nebium-Small", 117e6, 2.3e9, "#1f77b4"),
        ("Nebium-Medium", 345e6, 6.9e9, "#ff7f0e"),
        ("Nebium-Large", 762e6, 15.2e9, "#d62728"),
    ]

    for name, p_count, tokens, col in tiers:
        L_point = chinchilla_loss(np.array([p_count]), tokens)[0]
        ax.scatter([p_count / 1e6], [L_point], color=col, s=80, zorder=6, edgecolor="black", linewidth=1.0)
        ax.annotate(
            f"{name}\n({p_count/1e6:.0f}M)",
            xy=(p_count / 1e6, L_point),
            xytext=(p_count / 1e6 * 1.15, L_point + 0.05),
            fontsize=8.5,
            fontweight="semibold",
            color=col,
        )

    # Mark current empirical observation
    if not math.isnan(empirical_loss) and empirical_loss > 0:
        ax.scatter([empirical_params / 1e6], [empirical_loss], color="#9467bd", s=140, marker="*", zorder=7, edgecolor="black", linewidth=1.2, label=f"Empirical: {current_tier}")
        ax.annotate(
            f"Observed: {empirical_loss:.4f} nats",
            xy=(empirical_params / 1e6, empirical_loss),
            xytext=(empirical_params / 1e6 * 0.5, empirical_loss + 0.12),
            arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.2),
            fontsize=9,
            fontweight="bold",
            color="#9467bd",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f3e5f5", ec="#ab47bc", lw=0.8),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters ($N$, Millions)", fontsize=10)
    ax.set_ylabel("Expected Cross-Entropy Loss (nats)", fontsize=10)
    ax.set_title("Nebium Family — Chinchilla Scaling Law Trajectory", fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, which="both", ls="--")
    ax.legend(loc="upper right", fontsize=9)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
    except Exception:
        pass
    return fig


def save_publication_figures(
    history: list[dict[str, Any]],
    empirical_params: int,
    empirical_loss: float,
    tier_name: str = "Nebium-Small",
    output_dir: str | Path = "paper_assets",
) -> dict[str, Path]:
    """
    Renders and saves high-resolution PNG and vector PDF copies of all
    publication figures to the specified output directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    # Figure 1: Training dynamics
    if history:
        fig_dyn = plot_training_dynamics(history, tier_name=tier_name)
        p1 = out_path / "training_dynamics.png"
        fig_dyn.savefig(p1, dpi=300, bbox_inches="tight")
        fig_dyn.savefig(out_path / "training_dynamics.pdf", bbox_inches="tight")
        plt.close(fig_dyn)
        saved_files["training_dynamics"] = p1

        # Figure 2: Accuracy & legality
        fig_acc = plot_accuracy_and_legality(history, tier_name=tier_name)
        p2 = out_path / "accuracy_and_legality.png"
        fig_acc.savefig(p2, dpi=300, bbox_inches="tight")
        fig_acc.savefig(out_path / "accuracy_and_legality.pdf", bbox_inches="tight")
        plt.close(fig_acc)
        saved_files["accuracy_and_legality"] = p2

    # Figure 3: Scaling laws
    fig_scale = plot_scaling_law_alignment(
        empirical_params=empirical_params,
        empirical_loss=empirical_loss,
        current_tier=tier_name,
    )
    p3 = out_path / "scaling_laws_alignment.png"
    fig_scale.savefig(p3, dpi=300, bbox_inches="tight")
    fig_scale.savefig(out_path / "scaling_laws_alignment.pdf", bbox_inches="tight")
    plt.close(fig_scale)
    saved_files["scaling_laws"] = p3

    return saved_files
