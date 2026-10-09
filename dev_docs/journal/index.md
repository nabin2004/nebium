# Engineering & Research Journal

The **Engineering Journal** tracks the chronological evolution of the Nebium research project. Each log captures a specific milestone, the problem faced, what was tried, what failed, and the key lessons learned.

---

## 📅 Chronological Entries

| # | Title | Core Topic | Key Insight |
|---|---|---|---|
| **01** | [Why a Causal Transformer for Chess?](01-why-causal-transformer.md) | Modeling Paradigm | Moving from MCTS / AlphaZero to self-supervised autoregressive next-token prediction. |
| **02** | [Tokenization & Spatial Geometry](02-tokenization-and-board-geometry.md) | Data Representation | Why space-separated UCI strings outperform 2D FEN matrices for autoregressive scaling. |
| **03** | [Scaling Laws & Compute Alignment](03-scaling-laws-and-chinchilla.md) | Model Sizing & Compute | Aligning Nebium family (8K to 762M) with Chinchilla compute budgets ($D \approx 20N$). |
| **04** | [Editable Intelligence & Activation Steering](04-editable-intelligence-experiments.md) | Interpretability & Editing | Probing whether rules are localized vs distributed; building NebiumScope for counterfactual edits. |

---

## 📝 Add a New Journal Entry

Use the standard template whenever you complete a coding session, run an experiment, or debug an issue:

[:material-file-document-plus: Open Dev Log Entry Template](template.md){ .md-button .md-button--primary }
