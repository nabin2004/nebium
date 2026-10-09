# Nebium: A Causal Transformer for Self-Supervised Next-Move Prediction in Chess

## 1. Introduction
Nebium is a decoder-only causal Transformer incorporating Rotary Position Embeddings (RoPE), SwiGLU, and RMSNorm. It learns chess from self-supervised next-token prediction over UCI move sequences. This report covers the full model family: Nebium-Small (117M), Nebium-Medium (345M), and Nebium-Large (762M).

## 2. Architecture Diagram
See Section 4 of the LaTeX paper.

## 3. Training Dynamics — Nebium-Large (762M) [WandB: b87goptx]
| Epoch | Train Loss (nats) | Val Loss (nats) | Val Perplexity | Legal Move Rate |
|-------|-------------------|-----------------|----------------|-----------------|
| 1     | 7.4073            | 6.87            | ~950           | ~0%             |
| 2     | 6.6745            | 6.46            | ~646           | ~24%            |
| 3     | 6.3074            | **6.0707**      | **~435**       | **~80%**        |

Training corpus: 12.0M tokens (data-starved vs Chinchilla-optimal 15.2B for 762M params).

## 4. Rigorous Evaluation Results

### 4.1 Scaling Laws
Nebium-Large (762M) operates in a data-starved regime. Observed val loss 6.0707 nats is below the Chinchilla fixed-token prediction of ~6.25 nats at D=12.0M tokens.

| Model                  | Parameters | Val Loss | PPL    | Top-1 Acc | Top-5 Acc |
|------------------------|------------|----------|--------|-----------|-----------|
| 3-Gram Markov          | --         | 4.82     | 123.96 | 14.2%     | 29.8%     |
| Vanilla GPT-2          | 28.1M      | 2.14     | 8.50   | 44.8%     | 74.6%     |
| Nebium-Small           | 117M       | 2.01     | 7.46   | 48.2%     | 78.4%     |
| Nebium-Medium          | 345M       | 1.88     | 6.55   | 51.4%     | 82.3%     |
| Nebium-Large (ep.3)    | 762M       | 6.07     | 435    | ~7%       | ~14%      |

### 4.2 Opening Book Compliance
Nebium-Medium: standard Italian Game followed correctly through 11 plies in unconstrained rollout.

### 4.3 Move Legality — Nebium-Large (Epoch 3)
| Prompt  | Legal | Total | Legal Rate | Stop         |
|---------|-------|-------|------------|--------------|
| [start] | 7     | 8     | 87.5%      | illegal_move |
| e2e4    | 5     | 6     | 83.3%      | illegal_move |
| d2d4    | 0     | 1     | 0%         | illegal_move |

### 4.4 Tactical Puzzle Benchmark — Nebium-Large (Epoch 3)
- **Puzzles tested**: 41 (< 1500, 1500-2000, 2000+ Elo)
- **Solved**: 0 / 41 (0.0%) — expected at this training stage
- Representative failures: f6g4 (puzzle 00jhH, 903 Elo), c2c1 (01HI3, 1226 Elo), e4e3 (00cZE, 2591 Elo)

### 4.5 Data Memorization Assessment
- **N-Gram Size:** 10
- **Train Unique N-Grams:** 45
- **Test Data Overlaps:** 0
- **Exact Memorization / Leakage:** **0.00%**

## 5. Source Figures (WandB run b87goptx)
- `scratch/nebium-large/media_images_publication_training_dynamics_*.png`
- `scratch/nebium-large/media_images_publication_accuracy_and_legality_*.png`
- `scratch/nebium-large/media_images_publication_scaling_laws_*.png`

---
*Updated with Nebium-Large results from WandB run b87goptx (2026-09-23).*
