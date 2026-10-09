# ADR-004: Activation Steering for Counterfactual Rule Editing

**Date:** 2026-10-08  
**Status:** Accepted  
**Deciders:** Nabin  

---

## 1. Context & Problem Statement

To study "editable intelligence" on Nebium (e.g. changing pawn rules to allow backward steps), we needed a mechanism to intervene on model behavior.

Direct weight editing (e.g. ROME, MEMIT, or rank-1 MLP edits) has severe drawbacks:
- It permanently alters model parameters.
- It often causes catastrophic side effects (e.g. syntax breakdown, illegal move explosion).
- It is difficult to scale across many candidate rule changes.

---

## 2. Decision Outcome

**Chosen Solution:** **Reversible Activation Steering via Forward Hooks (`nebium_scope/interventions/steering.py`)**

Instead of updating weights $W$, we intercept hidden states at layer $L$ using PyTorch forward hooks:
$$h_L \leftarrow h_L + \alpha \cdot v$$

### Key Advantages:
1. **100% Reversible:** Managed via a clean Python context manager (`with steering:`), leaving base weights pristine.
2. **Continuous Modulation:** The scalar $\alpha$ provides a continuous "knob" from 0.0 (baseline standard chess) to 3.0 (strong counterfactual steering).
3. **Reproducible Concept Vectors:** Directions $v = \bar{h}_{\text{edited}} - \bar{h}_{\text{normal}}$ are serialized as `.pt` files and loaded on demand.
