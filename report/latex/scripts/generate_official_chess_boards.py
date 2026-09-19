import os
import sys
import chess
import chess.svg
import resvg_py
from PIL import Image
import io
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
out_dir = os.path.abspath(r'report/latex/figures')

# 1. Italian Game Main Line after 8. b4 Bb6
moves = 'e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 e1g1 g8f6 d2d3 e8g8 b1c3 d7d6 a2a3 h7h6 b2b4 c5b6'.split()
b1 = chess.Board()
for m in moves:
    b1.push_san(m)

# 2. Position after 11. c3
moves2 = 'c3d5 c6d4 f3d4 b6d4 c2c3'.split()
b2 = b1.copy()
for m in moves2:
    b2.push_san(m)

# 3. Position after 11... Kh8 12. cxd4
b3 = b2.copy()
b3.push_san('g8h8')
b3.push_san('c3d4')

svg1 = chess.svg.board(b1, lastmove=b1.peek(), arrows=[chess.svg.Arrow(chess.B2, chess.B4, color='#1b7837cc')], size=1200)
svg2 = chess.svg.board(b2, lastmove=b2.peek(), arrows=[chess.svg.Arrow(chess.C3, chess.D4, color='#d73027cc')], size=1200)
svg3 = chess.svg.board(b3, lastmove=b3.peek(), arrows=[chess.svg.Arrow(chess.G8, chess.H8, color='#d73027cc')], size=1200)

im1 = Image.open(io.BytesIO(resvg_py.svg_to_bytes(svg1)))
im2 = Image.open(io.BytesIO(resvg_py.svg_to_bytes(svg2)))
im3 = Image.open(io.BytesIO(resvg_py.svg_to_bytes(svg3)))

# Save individual PNGs
im1.save(os.path.join(out_dir, 'board_italian_1.png'))
im2.save(os.path.join(out_dir, 'board_italian_2.png'))
im3.save(os.path.join(out_dir, 'board_italian_3.png'))

# Create high-res composite figure with tight margins and larger boards
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 5.4), dpi=300, gridspec_kw={'wspace': 0.04})

for ax, im, title, note in [
    (ax1, im1, "(a) Italian Main Line: 8. b4 Bb6", "Opening development: castled king,\ncoordinated minor pieces."),
    (ax2, im2, "(b) Tactical Strike: 11. c3", "White strikes at the d4 bishop;\nbook response is 11... Bb6."),
    (ax3, im3, "(c) Autoregressive Drift: 11... Kh8?! 12. cxd4", "Failure mode: unprovoked king move;\nfree bishop conceded on d4.")
]:
    ax.imshow(im)
    ax.set_title(title, pad=8, fontsize=12, fontweight='bold', fontfamily='serif')
    ax.set_xlabel(note, fontsize=10, fontstyle='italic', fontfamily='serif', labelpad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.suptitle("Qualitative Trajectory: Italian Game Development and Autoregressive King Drift",
             fontsize=13.5, fontweight='bold', fontfamily='serif', y=0.99)
plt.tight_layout()

pdf_path = os.path.join(out_dir, "italian_game_qualitative.pdf")
png_path = os.path.join(out_dir, "italian_game_qualitative.png")

plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300, pad_inches=0.03)
plt.close()
print("Saved enlarged high-res composite italian_game_qualitative.pdf and png")
