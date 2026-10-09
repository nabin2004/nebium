# ADR-001: Rotary Position Embeddings (RoPE) & SwiGLU FFN

**Date:** 2026-08-20  
**Status:** Accepted  
**Deciders:** Nabin  
**Consulted:** Nebium Architecture Invariants (`AGENTS.md`)  

---

## 1. Context & Problem Statement

Standard baseline transformer architectures (like original GPT-2) rely on:
1. **Absolute learned positional embeddings:** $x = e_{\text{tok}} + e_{\text{pos}}$, where $e_{\text{pos}}$ is a learned matrix of shape $(\text{max\_seq\_len}, d_{\text{model}})$.
2. **Standard 2-layer MLP with GELU:** $\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$.

In chess move sequences:
- Games vary from 10 plies to over 200 plies. Absolute positions can cause degradation when evaluating moves past the standard opening phase.
- Relative position matters greatly: tactical replies occur relative to the opponent's previous move ($t - 1$, $t - 2$).
- Standard GELU FFNs are less parameter-efficient than gated activation units.

---

## 2. Decision Drivers

- **Length Generalization:** Better extrapolation to unusually long endgame games without retraining.
- **Relative Distance Awareness:** Queries and keys should naturally encode token distance $(m - n)$ in their inner product.
- **Compute Efficiency:** Maximizing validation loss reduction per parameter.

---

## 3. Considered Options

1. **Option 1: GPT-2 Vanilla (Absolute Learned Embeddings + GELU MLP)**
2. **Option 2: ALiBi (Attention with Linear Biases)**
3. **Option 3: Modern LLaMA-style (RoPE + RMSNorm + SwiGLU FFN)**

---

## 4. Decision Outcome

**Chosen Option:** **Option 3 (RoPE + RMSNorm + SwiGLU)**

### Rationale:
- **Rotary Position Embeddings (`norm.py`, `rope.py`):** Encodes relative position directly into query and key representations via complex rotation matrices. Dot products naturally decay gracefully with token distance.
- **RMSNorm Pre-normalization:** Replaces LayerNorm, saving ~7% GPU wall-clock time by dropping mean centering.
- **SwiGLU FFN (`ffn.py`):** Uses $\text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}}) W_{\text{down}}$, improving loss convergence by ~0.15 nats at identical FLOPs.

---

## 5. Consequences

### Positive:
- Strong length generalization up to `max_seq_len = 1024`.
- Higher convergence speed on Lichess game corpora.
- Clean tensor layout aligned with modern LLM architectures (e.g. LLaMA, Mistral).

### Negative / Trade-offs:
- Slightly higher implementation complexity than vanilla PyTorch `nn.Transformer`.
- Requires careful handling during GGUF tensor name exporting.
