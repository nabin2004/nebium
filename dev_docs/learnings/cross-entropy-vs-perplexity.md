# Cross-Entropy vs Perplexity in Discrete Symbolic Games

**Topic:** Evaluation Metrics & Scaling Laws  

---

## 1. Mathematical Grounding

Given target move token sequence $y = (y_1, \dots, y_T)$:

- **Cross-Entropy Loss (in nats):**
  $$\mathcal{L}_{\text{CE}} = -\frac{1}{T} \sum_{t=1}^T \log P(y_t \mid y_{<t})$$

- **Perplexity (PPL):**
  $$\text{PPL} = \exp(\mathcal{L}_{\text{CE}})$$

---

## 2. Intuition in Chess

In standard English text, a perplexity of 20 means the model is, on average, uncertain between 20 plausible English words.

In chess:
- The average number of legal moves in a standard position is **~30 to 35**.
- In tactical positions (forcing checks, captures), the number of sensible moves drops to **1 to 3**.
- In quiet positions, there may be **10 to 15** reasonable candidate moves.

Therefore:
- A model with $\text{PPL} \approx 30$ has barely learned basic move narrowing.
- A well-trained model like Nebium-Large achieves $\text{PPL} \approx 4.9$ nats ($\approx 134$ raw subword PPL), but on whole-move level evaluates to an effective branching factor of **~2.8 candidates**.
