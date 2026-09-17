# Automated Evaluation Pipeline

Nebium uses a multi-layer evaluation pipeline that assesses chess capability far beyond next-token cross-entropy loss.

## The Master Script

`scripts/generate_paper_report.py` acts as the orchestrator. It runs all individual evaluation scripts, collects their JSON outputs, generates matplotlib charts (`paper_assets/`), and compiles a publication-ready `FINAL_PAPER_REPORT.md`.

This script is automatically executed at the end of the `train.py` loop if `cfg.training.generate_report` is enabled.

---

## Per-Epoch Inline Evaluation (`src/evaluation/chess_metrics.py`)

These run after **every epoch** directly inside `train.py`.

### Legal Move Rate (`generate_sample_games`)

The model generates moves from prompt positions with **zero external filtering** — raw token probabilities only.

Each generated move is validated by `python-chess`. As soon as an illegal move is generated, **generation stops immediately**:

```python
if move in board.legal_moves:
    board.push(move)
    legal_count += 1
else:
    stop_reason = "illegal_move"
    break  # board has diverged from token sequence; stop here
```

> **Important**: continuing after an illegal move would evaluate subsequent tokens against a stale board state and artificially inflate the legality metric. If the prompt itself contains invalid UCI, the sample is skipped entirely (zero result).

**Reported metric**: `val/legal_move_rate`

### Tactical Puzzle Accuracy (`evaluate_puzzles`)

- Puzzles loaded from `data/fixtures/puzzles.jsonl` (generated from Lichess puzzle CSV by `scripts/prepare_puzzles.py`).
- Each puzzle: `prompt` = full UCI move history to the critical position + opponent's blunder; `solution` = single best response.
- Model generates **one move** greedily (`temperature=0.0`); exact match with `solution` counts as correct.
- Accuracy stratified by Lichess rating bracket: `<1500`, `1500-2000`, `2000+`.

**Reported metrics**: `val/puzzle_accuracy`, `val/puzzle_acc_under_1500`, `val/puzzle_acc_1500_to_2000`, `val/puzzle_acc_2000_plus`

---

## Post-Training Evaluation Scripts

These run once at the end of training via `scripts/generate_paper_report.py`.

### 1. Self-Play Elo Estimation (`scripts/eval_self_play.py`)

Pits Nebium against Stockfish at a fixed search depth (default depth 10 ≈ 2000 Elo). Alternates colours across games. Elo is computed as:

```
Elo_Nebium = 2000 - 400 * log10(1/Score - 1)
```

Move selection: legal-filtered greedy — scans logits from highest to lowest probability and picks the first token that parses as a legal UCI move. Falls back to a random legal move if none found.

### 2. Opening Book Compliance (`scripts/eval_openings.py`)

Each entry maps an opening name to a `(prompt, expected_next_move)` pair. The model is given the prompt and must predict the expected move:

| Opening | Prompt | Expected |
|---|---|---|
| Ruy Lopez (3...a6) | `e2e4 e7e5 g1f3 b8c6 f1b5` | `a7a6` |
| Sicilian (2.Nf3) | `e2e4 c7c5` | `g1f3` |
| Queen's Gambit (2...dxc4) | `d2d4 d7d5 c2c4` | `d5c4` |
| French (2.d4) | `e2e4 e7e6` | `d2d4` |
| Caro-Kann (2.d4) | `e2e4 c7c6` | `d2d4` |
| Italian (3...Bc5) | `e2e4 e7e5 g1f3 b8c6 f1c4` | `f8c5` |

> Note: strict exact-match. A "wrong" answer may still be theoretically sound (transposition).

### 3. Endgame Conversion (`scripts/eval_endgames.py`)

Starts from simplified board positions, Nebium plays one side against Stockfish for up to 50 moves. Measures whether the model can convert a winning advantage or avoid losing.

### 4. Blunder and Loop Analysis (`scripts/analyze_errors.py`)

Evaluates game quality using Stockfish Centipawn evaluations before and after each Nebium move.

**Blunder**: evaluation drops >2.0 pawns from the moving side's perspective.

```python
# After board.push(), board.turn has flipped to the opponent.
# board.turn == chess.BLACK means White just moved.
white_just_moved = (board.turn == chess.BLACK)
if white_just_moved:
    drop = (score_before - score_after) / 100.0  # White wants score high
else:
    drop = (score_after - score_before) / 100.0  # Black wants score low
```

**Loop**: games ending in 3-fold repetition or 50-move rule — common failure modes of causal models lacking lookahead.

### 5. Memorization & Data Leakage (`scripts/analyze_memorization.py`)

Computes N-gram overlap between training corpus and puzzle/test datasets. Assesses generalization vs. memorization.

### 6. Inference Latency (`scripts/benchmark_latency.py`)

Benchmarks time-to-first-token for: Greedy, Top-K, Top-P (nucleus), and Legal-Filtered decoding. Legal-Filtered guarantees 100% legal moves by masking illegal tokens to −∞ before sampling.

