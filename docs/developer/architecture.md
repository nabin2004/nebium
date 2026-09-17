# Nebium Architecture

This document outlines the core architecture of the Nebium model, an autoregressive language model designed explicitly for chess move prediction.

## Overview
Nebium treats a chess game as a sequence of tokens, where each token corresponds to a UCI move (e.g., `e2e4`, `g1f3`). It uses a standard causal **Decoder-Only Transformer** architecture but incorporates several modern improvements.

## Key Components

### 1. Rotary Position Embeddings (RoPE)
Instead of standard absolute position embeddings (like sine/cosine or learned embeddings), Nebium uses **RoPE**. 
- **Why?** In chess, the relative distance between moves is highly informative (e.g., a maneuver spanning 3 moves vs. 15 moves). RoPE elegantly encodes relative positional information directly into the Query and Key vectors of the attention mechanism.
- **Location:** `src/models/transformer/modules.py` -> `apply_rotary_emb`

### 2. SwiGLU Activation Function
The Feed-Forward Network (FFN) inside the transformer blocks uses a **SwiGLU** activation instead of standard ReLU or GELU.
- **Why?** SwiGLU (Swish-Gated Linear Unit) has been shown to offer superior performance and convergence in language models (used in LLaMA, PaLM).
- **Location:** `src/models/transformer/modules.py` -> `SwiGLUFFN`

### 3. RMSNorm (Root Mean Square Normalization)
Standard Layer Normalization centers the activations and scales them. RMSNorm drops the mean-centering entirely, which is computationally cheaper and works just as well.
- **Why?** Improves training speed without sacrificing stability.
- **Location:** `src/models/transformer/modules.py` -> `RMSNorm`

### 4. Custom Vocabulary (BPE Tokenizer)
Unlike a standard English text tokenizer, our custom Byte-Pair Encoding tokenizer is trained strictly on UCI chess moves. 
- It efficiently tokenizes moves into singular tokens or tightly packed character clusters.
- The `vocab_size` typically resolves to around 5,000 for standard competitive games.

## Inference and Legal Move Masking

A core contribution of Nebium is the **Legal Move Masking Layer**.

Language models can hallucinate invalid tokens. To prevent illegal chess moves:
1. The model predicts a probability distribution over the entire vocabulary (logits).
2. During inference, we pass the generated token sequence to `python-chess`, which reconstructs the board state.
3. We extract all legal UCI moves for that exact board state.
4. We mask out any logit that corresponds to an illegal move (setting its probability to negative infinity).
5. We then sample (or greedy-decode) from the remaining legal tokens.

This ensures **100% legal play** during inference, even if the model predicts a blunder.

- **Location:** `src/models/transformer/nebium.py` -> `generate()` and `beam_search_generate()`
