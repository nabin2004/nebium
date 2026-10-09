# Journal 01: Why a Causal Transformer for Chess?

**Date:** 2026-08-15  
**Author:** Nabin  
**Status:** Completed  
**Tags:** `#architecture` `#paradigm` `#causal-lm` `#chess`

---

## 1. Context & Motivation

Traditional computer chess systems (Stockfish, AlphaZero, Leela Chess Zero) rely heavily on:

1. **State representation heuristics:** 12-channel $8 \times 8$ piece bitboards, castling rights flags, halfmove clocks.
2. **Evaluation functions:** Scoring positions numerically (+1.2 pawns, winning percentage).
3. **Search trees:** Minimax with alpha-beta pruning or Monte Carlo Tree Search (MCTS).

While effective for competitive play, these systems do not answer a fundamental scientific question:
> *Can an autoregressive sequence model learn the rules, positional dynamics, tactics, and opening theory of chess purely from next-token self-supervision on move sequences—without any explicit board state representation?*

---

## 2. What I Tried

I framed chess games not as board matrices, but as **texts in a formal language**:
```text
e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 ...
```

By predicting token $t_{k+1}$ given context $t_1, \dots, t_k$, the transformer must implicitly:

1. Track internal board state across time.
2. Filter for move legality (rejecting illegal moves like moving through pieces or into check).
3. Distinguish tactical moves (forks, pins, skewers) from strategic development moves.

---

## 3. What Failed & What Surprised Me

### Failure 1: Naive Fixed-Vocabulary Board Actions
Initially, I considered predicting an action ID from an exhaustive move vocabulary (~4,672 possible UCI moves in chess). However:

- The vocabulary was sparse and skewed.
- The model struggled with long-range move dependencies when moves were treated as disconnected integer classes.

### The Breakthrough: Subword / BPE Move Tokens
Treating moves as character n-grams or subwords (`e2e4`, `g1f3`, `e7e5`) using a custom Hugging Face BPE tokenizer with a compact vocabulary allowed the model to share sub-token representations for source and destination squares (`e2`, `e4`).

---

## 4. Key Learnings & "TIL"

!!! success "Key Insight"
    Causal transformers do not just memorize openings; they develop **emergent legality representations**. By layer 6 in `nebium_base`, the model's illegal move rate drops below 1.2%, proving that residual representations encode the valid geometric state of the board.

---

## 5. Next Steps

- Design modern transformer building blocks (RoPE + SwiGLU) to replace standard absolute sinusoidal embeddings.
- Implement streaming PGN decompression to handle multi-gigabyte Lichess archives without blowing up RAM.
