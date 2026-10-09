# ADR-002: Decoupled Training via Hydra Configurations

**Date:** 2026-09-02  
**Status:** Accepted  
**Deciders:** Nabin  

---

## 1. Context & Problem Statement

Machine learning research projects often suffer from "script drift": researchers hardcode hyperparameters, batch sizes, data paths, and logger options directly into `train.py`.

As Nebium grew to support multiple model tiers (`nebium_stub`, `nebium_base`, `nebium_medium`, `nebium_large`) and multiple deployment environments (local developer laptop, remote headless Kaggle GPU, CI test runners), hardcoded parameters led to bugs and non-reproducible runs.

---

## 2. Decision Outcome

**Chosen Solution:** **Hierarchical Hydra Configurations (`configs/`)**

We decoupled the training engine (`src/training/trainer.py`) from:
1. **Model architecture:** `configs/model/` (`stub.yaml`, `base.yaml`, `medium.yaml`, etc.)
2. **Dataset source:** `configs/data/` (`fixtures.yaml`, `lichess.yaml`)
3. **Training regime:** `configs/training/` (`default.yaml`, `smoke_test.yaml`)
4. **Telemetry logger:** `configs/logging/` (`wandb.yaml`, `tensorboard.yaml`, `disabled.yaml`)

```text
configs/
├── config.yaml
├── model/       # stub, base, small, medium, large
├── data/        # fixtures, lichess
├── training/    # default, fast, smoke_test
└── logging/     # wandb, tensorboard, disabled
```

---

## 3. Key Benefits

- **Zero hardcoding:** Developers can override any parameter via CLI:
  ```bash
  python scripts/train.py model=nebium_base training.batch_size=32 logging=disabled
  ```
- **Hermetic Unit & Smoke Tests:** CI runs `pytest` using `model=nebium_stub` and `data=fixtures`, completing in under 5 seconds without network calls.
