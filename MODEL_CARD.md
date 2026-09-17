---
language: en
tags:
- chess
- transformer
- autoregressive
---

# Nebium Chess Transformer

Nebium is a small autoregressive transformer model designed to play chess by predicting standard UCI moves directly from move history. It is trained entirely in a self-supervised manner on a dataset of chess games.

## Model Details

- **Architecture:** Transformer (Autoregressive, similar to GPT)
- **Positional Encoding:** RoPE (Rotary Position Embeddings)
- **Tokenization:** Custom BPE tokenization over UCI move strings (e.g. `e2e4`, `g1f3`)
- **Capabilities:** Next-move prediction, Beam Search decoding, Illegal move filtering.

## Evaluation

The model is evaluated across multiple dimensions at each epoch and comprehensively after training:

- **Legal Move Rate**: The model generates moves with zero external filtering (raw token logits only). Each move is validated by `python-chess`; generation stops at the first illegal move, since continuing would corrupt the board state and inflate the metric. Reported as `val/legal_move_rate`.
- **Tactic Puzzles**: Evaluated greedily on Lichess puzzles reconstructed as full UCI move histories, stratified by Elo bracket (`<1500`, `1500–2000`, `2000+`).
- **Elo Estimation**: Bayesian Elo computed via self-play against Stockfish Depth 10 (~2000 Elo). Move selection is legal-filtered greedy (highest-probability legal token).
- **Opening Book Compliance**: Tested against 6 mainline openings with explicit `(prompt → expected_next_move)` pairs sourced from theory.
- **Blunder Rate**: Measured via Stockfish Centipawn evaluation drop (>2.0 pawns = blunder) with correct per-side sign convention.

## Usage

You can use the model using the provided Python scripts.

```python
# API example using FastAPI
import requests
resp = requests.post("http://127.0.0.1:8000/predict", json={"moves": "e2e4 e7e5", "temperature": 0.0})
print(resp.json())
```

## Known Limitations
- The raw model (without legal move filtering) will stop generating once it produces an illegal move — the game position and token sequence diverge at that point.
- It relies entirely on move history, so context is bounded by `max_seq_len`. Very long games (>512 tokens) are truncated from the left.
- Opening compliance uses strict exact-match; transpositions that are equally valid in theory count as failures.

