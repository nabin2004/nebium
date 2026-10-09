# Journal 02: Tokenization & Spatial Board Geometry

**Date:** 2026-08-28  
**Author:** Nabin  
**Status:** Completed  
**Tags:** `#tokenization` `#data-pipeline` `#bpe` `#uci`

---

## 1. Context & Motivation

In NLP, tokenization choices (WordPiece vs BPE vs Byte-level) dramatically affect subword boundary artifacts and sequence lengths. In chess, the choice of move serialization format dictates how easily attention heads can form square-to-square geometric circuits.

I needed to decide between three representations:
1. **FEN snapshots per ply:** e.g. `rnbqkbnr/pppppppp/... w KQkq - 0 1`
2. **Standard Algebraic Notation (SAN):** e.g. `1. e4 e5 2. Nf3 Nc6`
3. **Pure Universal Chess Interface (UCI):** e.g. `e2e4 e7e5 g1f3 b8c6`

---

## 2. The Trade-Off Matrix

| Format | Ambiguity | Sequence Length | Context Overhead |
|---|---|---|---|
| **FEN per ply** | None | Extremely long (70+ tokens per move) | Redundant; 90% of squares do not change per ply |
| **SAN (`Nf3`, `exd5`)** | Context-dependent (requires disambiguation like `Nbd7` vs `Nfd7`) | Compact (~3-5 chars per move) | High parsing complexity; ambiguous piece origin |
| **UCI (`g1f3`, `e4d5`)** | **Zero ambiguity**; source square and destination square explicitly declared | **Uniform** (4-5 chars per move) | Cleanest tokenization for sequence models |

---

## 3. What I Tried & What Failed

### Experiment: Training BPE with Space vs No-Space Separation
If games are formatted as continuous strings: `e2e4e7e5g1f3b8c6`, BPE creates unnatural splits across move boundaries (e.g. `4e7`, `5g1`).

### Solution: Space-Delimited BPE with Pre-tokenization
By strictly splitting on whitespace (`e2e4 e7e5 g1f3`), each move is guaranteed to be parsed as a self-contained token or two 2-character square tokens:
- Source coordinate: `e2`
- Destination coordinate: `e4`
- Optional promotion character: `q`

---

## 4. Key Learnings & "TIL"

!!! tip "Why UCI is superior for Transformers"
    Because UCI explicitly declares `[source_square][dest_square]`, self-attention heads can directly attend between the source square token and the destination square token. An attention weight matrix can therefore learn the legal transformation map directly!

---

## 5. Artifacts Created

- Custom BPE tokenizer trained via Hugging Face `tokenizers` library (`src/data/tokenizer.py`).
- Fast streaming PGN parser filtering games with minimum Elo and move counts (`src/data/prepare.py`).
