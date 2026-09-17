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

The model has been evaluated extensively:
- **Tactic Puzzles:** Evaluated on Lichess puzzle datasets, stratified by Elo rating brackets.
- **Elo Estimation:** Estimated through self-play against Stockfish (Depth 10).
- **Opening Book Compliance:** Tested against standard main lines like Ruy Lopez, Sicilian, and Queen's Gambit.

## Usage

You can use the model using the provided Python scripts.

```python
# API example using FastAPI
import requests
resp = requests.post("http://127.0.0.1:8000/predict", json={"moves": "e2e4 e7e5", "temperature": 0.0})
print(resp.json())
```

## Known Limitations
- Without the legal move filtering layer, the model may hallucinate illegal moves, especially in complex endgames or long sequences.
- It relies completely on the move history, which bounds its context to the maximum sequence length.
