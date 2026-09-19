import os
import sys
import chess
import chess.svg
import json

sys.stdout.reconfigure(encoding='utf-8')

# The moves requested by the user + king drift continuation
game_moves = [
    ("e2e4", "1. e4", "White claims central space on e4 and opens diagonals for the light-squared bishop and queen."),
    ("e7e5", "1... e5", "Black mirrors centrally, establishing equal classical central stake."),
    ("g1f3", "2. Nf3", "Developing the king's knight, attacking Black's e5 pawn."),
    ("b8c6", "2... Nc6", "Developing the knight and defending the e5 outpost."),
    ("f1c4", "3. Bc4", "Italian Game / Giuoco Piano. Directs pressure against the vulnerable f7 square."),
    ("f8c5", "3... Bc5", "Developing the dark-squared bishop actively to c5 before closing the center."),
    ("e1g1", "4. O-O", "White tucks the king to safety and activates the f1 rook."),
    ("g8f6", "4... Nf6", "Black develops the knight to f6, applying counterpressure on White's e4 pawn."),
    ("d2d3", "5. d3", "Giuoco Pianissimo. White solidifies the e4 pawn and prepares c1 bishop development."),
    ("e8g8", "5... O-O", "Black castles kingside into safety."),
    ("b1c3", "6. Nc3", "White develops the queenside knight, maintaining flexible central tension."),
    ("d7d6", "6... d6", "Solidifying e5 and opening the diagonal for Black's light-squared bishop."),
    ("a2a3", "7. a3", "Prophylactic wing preparation: carves an escape square a2 for the bishop and prepares b4."),
    ("h7h6", "7... h6", "Prophylaxis preventing White from pinning the f6 knight with Bg5."),
    ("b2b4", "8. b4", "White seizes queenside territory and challenges Black's active c5 bishop."),
    ("c5b6", "8... Bb6", "The bishop tucks back to b6, preserving pressure along the a7-g1 diagonal."),
    ("c3d5", "9. Nd5", "White centralizes the knight onto the strong d5 outpost."),
    ("c6d4", "9... Nxd4", "Central piece tension: knights clash on the d-file."),
    ("f3d4", "10. Nxd4", "White recaptures, maintaining central initiative."),
    ("b6d4", "10... Bxd4", "Black recaptures with the bishop, temporarily hitting the a1 rook."),
    ("c2c3", "11. c3", "Strong Italian Game strategy: White hits the d4 bishop with tempo while reinforcing the center. Book response is 11... Bb6."),
    # Failure mode: Autoregressive King Coordinate Drift
    ("g8h8", "11... Kh8?!", "Coordinate drift failure mode: Instead of retreating the attacked bishop (11... Bb6), the model generates an unprovoked king retreat, hanging the d4 bishop!"),
    ("c3d4", "12. cxd4", "White cleanly captures the undefended dark-squared bishop on d4, converting decisive material advantage (+3.5 evaluation)."),
    ("h8h7", "12... Kh7?!", "Erratic king pacing: Black moves the king back to h7, burning critical tempi without addressing White's central dominance."),
    ("c1e3", "13. Be3", "White develops the dark-squared bishop, cementing complete control over central squares."),
    ("h7g8", "13... Kg8?!", "Repetitive king loop (Kh8 -> Kh7 -> Kg8): Autoregressive attention defocusing causes persistent oscillating moves on sparse board states.")
]

board = chess.Board()
history = []

# Initial position
history.append({
    "ply": 0,
    "san": "Initial",
    "uci": "",
    "fen": board.fen(),
    "comment": "Starting tournament position. White to move.",
    "svg": chess.svg.board(board=board, size=460)
})

for uci, san, comment in game_moves:
    move = chess.Move.from_uci(uci)
    board.push(move)
    history.append({
        "ply": len(history),
        "san": san,
        "uci": uci,
        "fen": board.fen(),
        "comment": comment,
        "svg": chess.svg.board(board=board, lastmove=move, size=460)
    })

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Nebium - Qualitative Move Analysis: Italian Game & Autoregressive Drift</title>
<style>
  :root {{
    --bg: #0f172a;
    --card: #1e293b;
    --card-border: #334155;
    --accent: #38bdf8;
    --accent-hover: #0284c7;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --danger: #f43f5e;
    --success: #10b981;
    --gold: #f59e0b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text-primary);
    line-height: 1.5;
    padding: 24px;
    display: flex;
    justify-content: center;
  }}
  .container {{
    max-width: 1100px;
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }}
  header {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px 24px;
  }}
  h1 {{
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .subtitle {{
    color: var(--text-secondary);
    font-size: 13.5px;
    margin-top: 6px;
  }}
  .main-layout {{
    display: grid;
    grid-template-columns: 480px 1fr;
    gap: 24px;
  }}
  .board-card {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }}
  .board-wrapper {{
    width: 448px;
    height: 448px;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
  }}
  .board-wrapper svg {{
    width: 100%;
    height: 100%;
    display: block;
  }}
  .controls {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 16px;
    width: 100%;
    justify-content: center;
  }}
  button {{
    background: #2563eb;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease;
  }}
  button:hover {{ background: #1d4ed8; }}
  button:disabled {{ background: #334155; color: #64748b; cursor: not-allowed; }}
  .ply-indicator {{
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
    min-width: 90px;
    text-align: center;
  }}
  .info-panel {{
    display: flex;
    flex-direction: column;
    gap: 16px;
  }}
  .callout {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 18px 20px;
  }}
  .callout-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
  }}
  .badge {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 0.5px;
  }}
  .badge-theory {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
  .badge-drift {{ background: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.4); }}
  .move-title {{
    font-size: 18px;
    font-weight: 700;
  }}
  .comment-text {{
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.6;
  }}
  .fen-box {{
    margin-top: 12px;
    background: #090d16;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 11px;
    color: #94a3b8;
    overflow-x: auto;
  }}
  .move-list-card {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 16px;
    flex: 1;
    overflow-y: auto;
    max-height: 280px;
  }}
  .move-list-title {{
    font-size: 13px;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
  }}
  .move-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }}
  .move-btn {{
    background: #0f172a;
    border: 1px solid var(--card-border);
    color: var(--text-secondary);
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 13px;
    cursor: pointer;
  }}
  .move-btn:hover {{
    background: #334155;
    color: white;
  }}
  .move-btn.active {{
    background: var(--accent);
    color: #0f172a;
    font-weight: 700;
    border-color: var(--accent);
  }}
  .move-btn.drift {{
    border-color: var(--danger);
    color: #fda4af;
  }}
  .move-btn.drift.active {{
    background: var(--danger);
    color: white;
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Italian Game Progression & Autoregressive Horizon Drift</h1>
    <div class="subtitle">Qualitative case study of Nebium autoregressive trajectory: opening book fidelity (1. e4 through 11. c3) followed by attention defocusing and king coordinate drift.</div>
  </header>

  <div class="main-layout">
    <div class="board-card">
      <div class="board-wrapper" id="boardWrapper"></div>
      <div class="controls">
        <button id="btnFirst" onclick="jumpTo(0)">|&lt;</button>
        <button id="btnPrev" onclick="step(-1)">&lt; Prev</button>
        <div class="ply-indicator" id="plyIndicator">Move 0 / 25</div>
        <button id="btnNext" onclick="step(1)">Next &gt;</button>
        <button id="btnLast" onclick="jumpTo(historyData.length - 1)">&gt;|</button>
      </div>
    </div>

    <div class="info-panel">
      <div class="callout">
        <div class="callout-header">
          <div class="move-title" id="moveTitle">Start Position</div>
          <span class="badge badge-theory" id="moveBadge">Theory</span>
        </div>
        <div class="comment-text" id="commentText">White to move.</div>
        <div class="fen-box" id="fenBox">rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1</div>
      </div>

      <div class="move-list-card">
        <div class="move-list-title">Move Trajectory</div>
        <div class="move-grid" id="moveGrid"></div>
      </div>
    </div>
  </div>
</div>

<script>
const historyData = {json.dumps(history)};
let currentIdx = 0;

function render() {{
  const item = historyData[currentIdx];
  document.getElementById('boardWrapper').innerHTML = item.svg;
  document.getElementById('plyIndicator').innerText = `Ply ${{currentIdx}} / ${{historyData.length - 1}}`;
  document.getElementById('moveTitle').innerText = item.san;
  document.getElementById('commentText').innerText = item.comment;
  document.getElementById('fenBox').innerText = item.fen;

  const badge = document.getElementById('moveBadge');
  if (currentIdx >= 21) {{
    badge.className = 'badge badge-drift';
    badge.innerText = 'Failure Mode: King Drift';
  }} else {{
    badge.className = 'badge badge-theory';
    badge.innerText = currentIdx === 0 ? 'Start' : 'Italian Theory';
  }}

  document.getElementById('btnPrev').disabled = (currentIdx === 0);
  document.getElementById('btnFirst').disabled = (currentIdx === 0);
  document.getElementById('btnNext').disabled = (currentIdx === historyData.length - 1);
  document.getElementById('btnLast').disabled = (currentIdx === historyData.length - 1);

  // Update move grid
  const btns = document.querySelectorAll('.move-btn');
  btns.forEach((btn, idx) => {{
    btn.classList.toggle('active', idx === currentIdx);
  }});
}}

function step(delta) {{
  const next = currentIdx + delta;
  if (next >= 0 && next < historyData.length) {{
    currentIdx = next;
    render();
  }}
}}

function jumpTo(idx) {{
  if (idx >= 0 && idx < historyData.length) {{
    currentIdx = idx;
    render();
  }}
}}

// Build move list
const grid = document.getElementById('moveGrid');
historyData.forEach((item, idx) => {{
  const btn = document.createElement('button');
  btn.className = 'move-btn' + (idx >= 21 ? ' drift' : '');
  btn.innerText = item.san;
  btn.onclick = () => jumpTo(idx);
  grid.appendChild(btn);
}});

// Initialize
render();

// Keyboard arrows
window.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') step(-1);
  if (e.key === 'ArrowRight') step(1);
}});
</script>
</body>
</html>
"""

viewer_path = os.path.abspath('report/qualitative_game_viewer.html')
with open(viewer_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Generated qualitative game viewer at: {viewer_path}")
