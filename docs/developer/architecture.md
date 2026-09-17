# Nebium Architecture

This document outlines the core architecture of the Nebium model, an autoregressive transformer
designed explicitly for chess move prediction.

---

## Overview

Nebium treats a chess game as a flat sequence of UCI move tokens (e.g., `e2e4`, `g1f3`). It uses
a causal **Decoder-Only Transformer** trained to predict the next move given all prior moves — a
standard next-token cross-entropy objective applied to a domain where the token vocabulary is chess
moves rather than natural language words.

The model is trained on PGN game files (configurable via `data.urls`). Games shorter than
`data.min_moves` (default: 10 half-moves) are discarded. The corpus is serialised as one
space-separated UCI move sequence per line before tokenisation.

---

## Key Components

### 1. Rotary Position Embeddings (RoPE)

Instead of absolute position embeddings, Nebium uses **RoPE**.

- **Why?** The relative distance between moves is semantically meaningful (a 3-move manoeuvre vs. a
  15-move manoeuvre carry different weight). RoPE encodes relative positional information directly
  into the Query and Key vectors of the attention mechanism, with no additional parameter cost.
- **Location:** `src/models/transformer/rope.py` → `apply_rotary_emb`

### 2. SwiGLU Activation Function

The Feed-Forward Network (FFN) inside each transformer block uses **SwiGLU** instead of ReLU or GELU.

- **Why?** SwiGLU (Swish-Gated Linear Unit) consistently improves convergence in language models
  (LLaMA, PaLM) and has the same theoretical cost as a standard two-layer FFN.
- **Location:** `src/models/transformer/ffn.py` → `SwiGLUFFN`

### 3. RMSNorm

Standard LayerNorm both centres and scales activations. **RMSNorm** drops the mean-centering step,
which is computationally cheaper and works equally well in practice.

- **Why?** Marginally faster training without sacrificing stability.
- **Location:** `src/models/transformer/norm.py` → `RMSNorm`

### 4. Tokeniser: BPE over UCI Moves

Nebium uses a **BPE tokeniser** from the HuggingFace `tokenizers` library, trained exclusively on
UCI chess move sequences.

#### How it actually works

The tokeniser uses a `Whitespace` pre-tokeniser, which splits the input on spaces _before_ BPE
applies. Because the corpus is formatted as space-separated moves (`"d2d4 d7d5 c2c4 ..."`), each
move string is delivered to the BPE algorithm as its own isolated "word". BPE then learns
sub-character merges _within_ each move string.

In practice, with `vocab_size ≈ 5000` trained on a corpus where move strings are 4–5 characters,
**almost all moves end up as a single token** — the BPE vocabulary is large enough to represent
every common UCI move as one ID. However, **this is not guaranteed by design**: BPE could in
principle represent an infrequent or promotion move as two character-cluster tokens. The
`legal_tokens_fn` callback in `generate()` accounts for this by supplying allowed token IDs for
the current board state at every generation step.

#### Special tokens

| Token | Purpose |
|-------|---------|
| `[PAD]` | Padding to a fixed sequence length |
| `[UNK]` | Unknown characters (rare in practice) |
| `[BOS]` | Beginning of game sequence |
| `[EOS]` | End of game sequence |
| `[SEP]` | Reserved separator |

#### Special-case UCI encoding

- **Castling**: encoded as king destination — `e1g1` (White kingside), `e1c1` (White queenside),
  `e8g8`, `e8c8`. These are ordinary 4-character strings and each becomes a single token.
- **En passant**: encoded as a normal diagonal pawn capture (e.g., `e5d6`). No special notation;
  `python-chess` infers en passant legality from board state.
- **Promotion**: a fifth character denotes the piece — `e7e8q`, `e7e8r`, `e7e8b`, `e7e8n`. These
  5-character strings are rare enough that BPE may split them into two tokens; the legal-move mask
  handles them correctly regardless.

#### Context length

`max_seq_len` (default: **128 tokens**) sets the model's context window. Since ≈1 token ≈ 1 move,
this covers roughly 64 full moves (128 half-moves / plies) per game. Games longer than this are
windowed to the most recent 128 tokens at inference time.

---

## Training Objective

Nebium is trained with **causal language modelling** — standard next-token cross-entropy over the
move sequence:

```python
loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), tokens[:, 1:].reshape(-1))
```

Each position predicts the next move. There is no auxiliary objective (value head, policy gradient)
in the default configuration.

---

## Inference and Legal Move Masking

### Why masking is needed

A language model samples from a distribution over its entire vocabulary — including tokens that
happen to correspond to illegal moves in the current position. Without intervention, the model can
produce moves that are syntactically valid UCI strings but board-state-invalid (e.g., moving a
pinned piece, moving out of check incorrectly).

### How the masking layer works

At each generation step:

1. The model produces logits over the full vocabulary.
2. `python-chess` reconstructs the board state from the token sequence generated so far.
3. All legal UCI moves for that position are enumerated.
4. Any token **not** in the legal-move set has its logit set to `−∞`.
5. Sampling or greedy decoding proceeds over the surviving logits.

- **Location:** `src/models/transformer/nebium.py` → `generate()` and `beam_search_generate()`

### Beam search and masking interaction

`beam_search_generate()` maintains `beam_width` candidate sequences in parallel. The legal-move
mask is applied **at every beam expansion step** — not just once at the start — because the set of
legal moves changes with each half-move. For each candidate sequence, `python-chess` independently
computes legal moves from that beam's board state, then the corresponding logits are masked before
top-k candidates are selected for that beam.

### Limitation: legality ≠ quality

> **Masking guarantees legal play; it does not guarantee good play.**

The masking layer redistributes probability mass from illegal tokens onto legal ones, but does
**not** re-rank legal moves. If the model assigns high confidence to a losing blunder, masking
leaves that blunder as the top choice — it is simply legal. A model can achieve 100% legal-move
rate while still playing at a very low Elo.

This means two evaluation axes must be tracked separately:

- **Legal-move rate** — does the model stay within board rules? (`chess_metrics.py`)
- **Move quality** — does the model play moves a strong engine would choose? (puzzle accuracy,
  Elo estimation via `eval_self_play.py`)

---

## Architecture Summary

| Hyperparameter | Default |
|----------------|---------|
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_layers` | 1 |
| `dropout` | 0.1 |
| `max_seq_len` | 128 tokens |
| `vocab_size` | 5000 |
| `positional_encoding` | RoPE |
| `activation` | SwiGLU |
| `norm` | RMSNorm |
| `attention_type` | standard (causal) |
| `tie_word_embeddings` | false |
| `bias` | false |

These are the baseline defaults in `src/config/schema.py`. Production runs use larger configs
defined in `conf/model/`.
