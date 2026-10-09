# Training Stability & Mitigating Loss Spikes in Transformer Training

**Topic:** Optimization & Pretraining Dynamics  

---

## The Symptom

During pretraining on the Lichess 2013 dataset (millions of moves), we occasionally observed severe loss spikes: validation loss jumping from 4.9 nats to >8.5 nats in a single step, followed by gradient NaN or unrecoverable parameter divergence.

---

## Root Causes Identified

1. **Unclipped Attention Logits:** In RoPE-augmented attention, extreme relative distance rotations occasionally produced query-key inner products $> 80$, causing softmax overflow.
2. **Abrupt Learning Rate Decay:** Stepping the learning rate too fast without cosine warmup destabilized the SwiGLU gating projections.
3. **Corrupt PGN Move Sequences:** Certain rare PGN variants contained multi-character comments or non-standard clock annotations that slipped past regex filters.

---

## Solutions Implemented

1. **Cosine Annealing with Warmup:** Added a 2,000-step linear warmup followed by cosine decay down to 10% of maximum LR.
2. **Gradient Norm Clipping:** Strict clipping to `max_grad_norm = 1.0` in `src/training/trainer.py`.
3. **Ignore Index on Padding:** Strictly enforcing `ignore_index = -100` in cross-entropy loss so that padded tokens never contribute to gradient backpropagation.
