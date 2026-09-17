# Training Pipeline

The training pipeline transforms raw PGN files into a trained Transformer model. It relies on `Hydra` for hyperparameter management and heavily utilizes PyTorch's native `autocast` for mixed-precision training.

## 1. Data Preparation

**Files:** `src/data/prepare.py`, `src/data/dataset.py`

Before training, raw PGN (Portable Game Notation) files must be converted into purely UCI (Universal Chess Interface) move strings.
- Only games meeting `min_elo` and `min_moves` criteria are kept.
- A Hugging Face `Tokenizer` (Byte-Pair Encoding) is trained on this text corpus.
- This creates the tokenizer vocabulary used to embed tokens.

## 2. Model Instantiation

**Files:** `scripts/train.py`

Hydra manages the configuration. `train.py` loads `configs/config.yaml` and merges it with a specific model profile (e.g., `configs/model/nebium_117m.yaml`).
- The `vocab_size` is dynamically injected into the configuration object from the trained tokenizer.
- The model is instantiated via `hydra.utils.instantiate`.

## 3. The Training Loop

**Files:** `src/training/loop.py`

The core loop `run_training` implements:
- **Gradient Accumulation**: To simulate large batch sizes on memory-constrained GPUs.
- **Mixed Precision (AMP)**: Uses `torch.amp.autocast` for rapid `fp16`/`bf16` training without gradient underflow (via `GradScaler`).
- **Learning Rate Scheduler**: A Cosine Annealing scheduler with warmup.
- **In-Training Evaluation**:
  - Periodically computes Validation Loss and Perplexity.
  - Samples physical games from the model using prompt stubs (e.g., `e2e4`) and computes the raw legal move rate.
  - Optionally solves a small suite of tactic puzzles to monitor behavioral progression.

## 4. Checkpointing

The loop maintains two checkpoints:
- `checkpoint.pt`: The latest epoch state.
- `best_model.pt`: The state with the lowest validation loss.

After training concludes, the script proceeds to GGUF export and Hugging Face Hub integration if enabled in the configuration.
