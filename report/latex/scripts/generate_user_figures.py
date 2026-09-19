import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import chess

sys.stdout.reconfigure(encoding='utf-8')

# Ensure directories
out_dir = os.path.abspath(r'report/latex/figures')
os.makedirs(out_dir, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11,
})

def draw_single_board(ax, board: chess.Board, title: str, arrow=None, highlight_sqs=None, note=None):
    """Draws a tournament-styled vector chessboard on the specified axes."""
    c_light = '#f0d9b5'
    c_dark = '#b58863'

    for r in range(8):
        for f in range(8):
            sq = chess.square(f, r)
            is_light = (r + f) % 2 == 1
            sq_color = c_light if is_light else c_dark
            if highlight_sqs and sq in highlight_sqs:
                sq_color = '#baca44' if is_light else '#8ba335'
            ax.add_patch(plt.Rectangle((f, r), 1, 1, color=sq_color, ec='none'))

            p = board.piece_at(sq)
            if p:
                sym = p.unicode_symbol()
                if p.color == chess.WHITE:
                    txt = ax.text(f + 0.5, r + 0.5, sym, ha='center', va='center', fontsize=20, fontfamily='DejaVu Sans', color='#ffffff', zorder=4)
                    txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground='#222222')])
                else:
                    ax.text(f + 0.5, r + 0.5, sym, ha='center', va='center', fontsize=20, fontfamily='DejaVu Sans', color='#1a1a1a', zorder=4)

    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    for i, file_char in enumerate(files):
        ax.text(i + 0.5, -0.3, file_char, ha='center', va='center', fontsize=8, color='#444444')
    for i in range(8):
        ax.text(-0.3, i + 0.5, str(i + 1), ha='center', va='center', fontsize=8, color='#444444')

    if arrow:
        f_from, r_from, f_to, r_to = arrow
        ax.annotate('', xy=(f_to + 0.5, r_to + 0.5), xytext=(f_from + 0.5, r_from + 0.5),
                    arrowprops=dict(facecolor='#D55E00', edgecolor='#8c3800', width=2.8, headwidth=8.5, shrink=0.12, alpha=0.95),
                    zorder=6)

    ax.set_xlim(-0.5, 8.2)
    ax.set_ylim(-0.5, 8.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, pad=8, fontsize=9.5, fontweight='bold')
    if note:
        ax.text(4.0, -0.85, note, ha='center', va='top', fontsize=7.5, fontstyle='italic', color='#333333')

def generate_italian_game_boards():
    """Renders 3 panels showing the Italian Game trajectory and king coordinate drift."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 4.0), dpi=300)

    # Panel 1: Move 8 - Italian Game Flank & Center Setup (after 8. b4 Bb6)
    moves_p1 = "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 e1g1 g8f6 d2d3 e8g8 b1c3 d7d6 a2a3 h7h6 b2b4 c5b6".split()
    b1 = chess.Board()
    for m in moves_p1:
        b1.push_san(m)
    draw_single_board(
        ax1, b1,
        title="(a) Italian Main Line: 8. b4 Bb6",
        arrow=(1, 1, 1, 3), # b2 to b4
        highlight_sqs=[chess.B2, chess.B4, chess.B6],
        note="Giuoco Pianissimo development;\npieces coordinated, kings securely castled."
    )

    # Panel 2: Move 11 - White plays 11. c3 attacking Black's Bishop on d4
    moves_p2 = "c3d5 c6d4 f3d4 b6d4 c2c3".split()
    b2 = b1.copy()
    for m in moves_p2:
        b2.push_san(m)
    draw_single_board(
        ax2, b2,
        title="(b) Tactical Tension: 11. c3",
        arrow=(2, 1, 2, 2), # c2 to c3 attacking d4
        highlight_sqs=[chess.C2, chess.C3, chess.D4],
        note="White strikes at the d4 bishop;\nstandard book retreat is 11... Bb6."
    )

    # Panel 3: Move 11-12 - Black King Drift: 11... Kh8?! followed by 12. cxd4
    b3 = b2.copy()
    b3.push_san("g8h8") # King retreat
    b3.push_san("c3d4") # White takes free bishop
    draw_single_board(
        ax3, b3,
        title="(c) Autoregressive Drift: 11... Kh8 12. cxd4",
        arrow=(6, 7, 7, 7), # g8 to h8 (King drift)
        highlight_sqs=[chess.G8, chess.H8, chess.D4],
        note="Spatial drift failure mode: King steps away;\nfree bishop conceded on d4."
    )

    plt.suptitle("Qualitative Trajectory: Italian Game Development and Autoregressive King Drift",
                 fontsize=11.5, fontweight='bold', y=1.03)
    plt.tight_layout()
    pdf_path = os.path.join(out_dir, "italian_game_qualitative.pdf")
    png_path = os.path.join(out_dir, "italian_game_qualitative.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved italian_game_qualitative.pdf and png")

def generate_benchmarks_plot():
    """Generates a 2-panel chart from the CSV exports."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=300)

    # Left Panel: Puzzle Solve Rates by Elo Rating Bracket (WandB eval/puzzle_benchmarks)
    brackets = ['<1500\n(Club)', '1500–2000\n(Intermediate)', '2000+\n(Master)']
    epoch1_rates = [0.0, 0.0, 0.0]
    epoch20_rates = [5/30 * 100, 2/32 * 100, 1/20 * 100]  # 16.67%, 6.25%, 5.0%

    x = np.arange(len(brackets))
    width = 0.35

    rects1 = ax1.bar(x - width/2, epoch1_rates, width, label='Epoch 1', color='#999999', edgecolor='#666666')
    rects2 = ax1.bar(x + width/2, epoch20_rates, width, label='Epoch 20', color='#0072B2', edgecolor='#004d7a')

    ax1.set_ylabel('Puzzle Solve Rate (%)', fontweight='bold')
    ax1.set_title('(a) Tactical Puzzle Accuracy by Rating Bracket', pad=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(brackets)
    ax1.set_ylim(0, 22)
    ax1.legend(frameon=True, facecolor='#fbfbfb', edgecolor='#cccccc')
    ax1.grid(True, linestyle='--', alpha=0.4, axis='y')

    # Add data labels
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f'{h:.1f}%',
                     xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='#004d7a')

    # Right Panel: Move Legality across Generation Epochs (WandB eval/sample_generations)
    epochs = [1, 2, 3, 16, 17, 20]
    # Mean legal moves out of 20
    avg_legal_pct = [
        (5.55 / 12.55) * 100,  # Ep 1: ~44.2%
        (20.0 / 20.0) * 100,   # Ep 2: 100%
        (3.67 / 4.67) * 100,   # Ep 3: ~78.5%
        100.0,                 # Ep 16: 100%
        100.0,                 # Ep 17: 100%
        (19.92 / 20.0) * 100   # Ep 20: 99.6%
    ]

    ax2.plot(epochs, avg_legal_pct, marker='o', linewidth=2.0, markersize=6, color='#D55E00', label='Unconstrained Legality')
    ax2.axhline(100.0, color='#009E73', linestyle=':', linewidth=1.5, label='100% Legal Horizon')
    ax2.set_xlabel('Training Epoch', fontweight='bold')
    ax2.set_ylabel('Legal Move Rate (%)', fontweight='bold')
    ax2.set_title('(b) Autoregressive Horizon Legality Progression', pad=10, fontweight='bold')
    ax2.set_ylim(35, 105)
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(frameon=True, facecolor='#fbfbfb', edgecolor='#cccccc', loc='lower right')

    for ep, val in zip(epochs, avg_legal_pct):
        ax2.annotate(f'{val:.1f}%',
                     xy=(ep, val),
                     xytext=(0, -14 if val > 95 else 6), textcoords="offset points",
                     ha='center', va='bottom', fontsize=7.5, color='#8c3800')

    plt.suptitle("Empirical Evaluation: Tactical Solve Accuracy and Move Legality Horizons",
                 fontsize=11.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    pdf_path = os.path.join(out_dir, "puzzle_and_generation_benchmarks.pdf")
    png_path = os.path.join(out_dir, "puzzle_and_generation_benchmarks.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved puzzle_and_generation_benchmarks.pdf and png")

if __name__ == "__main__":
    generate_italian_game_boards()
    generate_benchmarks_plot()
