# Architecture Decision Records (ADRs)

An **Architecture Decision Record (ADR)** captures a significant architectural decision made in the project along with its context, alternatives considered, and consequences.

---

## 📋 ADR Index

| ADR ID | Title | Status | Date | Decision Summary |
|---|---|---|---|---|
| **[ADR-001](ADR-001-rope-and-swiglu.md)** | Rotary Position Embeddings (RoPE) & SwiGLU | **Accepted** | 2026-08-20 | Adopt RoPE in self-attention queries/keys and SwiGLU FFN instead of standard GPT-2 layers. |
| **[ADR-002](ADR-002-hydra-decoupled-training.md)** | Decoupled Training via Hydra | **Accepted** | 2026-09-02 | Decouple training engine from PGN streaming and W&B logging via hierarchical YAML configs. |
| **[ADR-003](ADR-003-gguf-for-inference.md)** | GGUF Quantization for Edge & CPU | **Accepted** | 2026-09-25 | Standardize local export on GGUF v3 to enable fast CPU/edge inference via llama.cpp. |
| **[ADR-004](ADR-004-activation-steering.md)** | Activation Steering for Counterfactual Rule Editing | **Accepted** | 2026-10-08 | Implement reversible activation steering via forward hooks rather than destructive weight edits. |

---

## 📝 Create a New ADR

To propose or document a new architectural decision:

[:material-file-document-edit: Open Blank ADR Template](adr-template.md){ .md-button .md-button--primary }
