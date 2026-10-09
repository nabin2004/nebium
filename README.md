# Nebium

A from-scratch causal Transformer for self-supervised next-move prediction on chess games. Configuration is Hydra YAML; the training loop talks to pluggable loggers and datasets, not to PGN or W&B directly.

### Published Models on Hugging Face
- **Nebium-Large (762M FP16 PyTorch):** [nabin2004/nebium-large](https://huggingface.co/nabin2004/nebium-large)
- **Nebium-Large (762M GGUF):** [nabin2004/nebium-large-gguf](https://huggingface.co/nabin2004/nebium-large-gguf)
- **Training Dataset:** [nabin2004/nebium-lichess-uci](https://huggingface.co/datasets/nabin2004/nebium-lichess-uci)

## Quick Install (Kaggle, Colab, or External Analysis)

To load Nebium models in Kaggle or any remote environment without cloning the full repository:

```bash
pip install git+https://github.com/nabin2004/nebium.git
```

```python
import nebium

# 1. Load pre-trained weights & tokenizer from Hugging Face:
model, tokenizer = nebium.load_model("large")  # Options: 'small', 'base', 'medium', 'large'

# 2. Predict next moves and top probabilities:
candidates = model.predict_next_moves("e2e4 e7e5 g1f3", tokenizer, top_k=5)
for move, prob in candidates:
    print(f"{move:<6} {prob * 100:.2f}%")

# 3. Generate autoregressive move continuations:
continuation = model.generate_moves("e2e4 e7e5 g1f3", tokenizer, max_new_moves=5)
print("Continuation:", continuation)

# 4. Or instantiate any model architecture from scratch:
model_stub = nebium.build_model("stub")    # 8K params
model_small = nebium.build_model("small")  # 117M params
model_large = nebium.build_model("large")  # 762M params
```

### Board Debugging & Spatial Features (`nebium.utils`)

The `nebium.utils` submodule provides utilities for validating UCI moves, inspecting spatial board features, rendering positions, and diagnosing model predictions:

```python
import nebium
from nebium import utils

# 1. Comprehensive diagnostic report on move sequences & candidate legality:
diag = utils.debug_position(
    prompt="e2e4 e7e5 g1f3",
    model=model,          # Optional: tests model's top predictions
    tokenizer=tokenizer,
    candidate_moves=["b8c6", "e8e7", "a1a8"], # Tests specific moves
    verbose=True,         # Prints ASCII board, material diff, legality & reasons
)

# 2. Extract 12-channel binary piece plane tensor (12, 8, 8) for probing/probing classifiers:
board = utils.get_board("e2e4 e7e5 g1f3")
tensor = utils.board_to_tensor(board)  # Shape: torch.Size([12, 8, 8])

# 3. Visualize position (renders rich SVG in Jupyter/Kaggle or ASCII in terminal):
utils.display_board("e2e4 e7e5 g1f3")

# 4. Move legality and board feature inspection:
print(utils.is_legal_move(board, "b8c6"))       # True
print(utils.get_legal_moves(board))             # List of all 29 legal UCI moves
print(utils.get_material_balance(board))        # Piece count & differential
print(utils.board_to_features(board))           # Turn, check, FEN, castling
```

### NebiumScope: Interpretability & Editable Intelligence (`nebium_scope`)

`nebium_scope` is an interactive interpretability toolkit for probing, visualizing, and steering chess rule representations (e.g., pawns moving backward, forward captures, knight diagonal leaps) on Nebium transformers without retraining.

Features:
- **Rule Engine & DSL:** Counterfactual move set generation with `VariantBoard` (`pawn_backward_one`, `pawn_capture_forward`, `knight_diagonal`, `super_pawn`).
- **Activation Steering:** Causal interventions ($h_{\ell}[t] \leftarrow h_{\ell}[t] + \alpha \cdot v$) using contrastive concept directions.
- **Logit Lens:** Layer-by-layer move probability projection through final norm and language model head.
- **Interactive Gradio Dashboard:** 4 tabs (Position Explorer, Causal Edit Lab, Logit Lens Inspector, Rule Benchmark Suite) with color-coded SVG chessboard arrows.

```python
import nebium_scope as ns

# 1. Launch the interactive Gradio Dashboard (http://localhost:7860)
ns.launch(server_port=7860)

# 2. Programmatic rule steering:
from nebium_scope.model import NebiumAdapter
from nebium_scope.interventions import ActivationSteering, ConceptVectorBuilder

adapter = NebiumAdapter(model, tokenizer)
vector = ConceptVectorBuilder.generate_mock_vector(d_model=adapter.d_model)

with ActivationSteering(adapter, layer=14, vector=vector, alpha=1.5):
    steered_moves = adapter.predict_top_k("e2e4 e7e5", k=5)
```

See [docs/NEBIUM_SCOPE.md](docs/NEBIUM_SCOPE.md) for the full developer guide, API reference, and experiment protocols.

### Developer Journey & Research Log (`dev_docs/`)

Nebium includes a comprehensive **Developer Journey & Engineering Log** documentation template built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/):
- **Live Local Preview:** `uv run mkdocs serve` (opens on `http://localhost:8000` with live reload)
- **Compile Production Site:** `uv run mkdocs build` (compiles to `docs/dev_journey/`)
- **Web Portal:** Access directly from the paper landing page at [docs/index.html](docs/index.html) or `docs/dev_journey/index.html`.

Includes:
- **Dev Journal:** Chronological entries on hypotheses, architectural pivots, and lessons learned.
- **Architecture Decision Records (ADRs):** Formal ADRs for RoPE, SwiGLU, Hydra decoupling, GGUF export, and activation steering.
- **Research Deep Dives:** Notes on attention heads, cross-entropy vs perplexity, and training stability.
- **Copy-Paste Templates:** Standardized templates for new dev entries and ADRs.

## Design principles


1. Configuration over code
2. Pluggable components
3. Dataset-agnostic interface
4. Training loop decoupling
5. Extensibility for future models

## Local Development Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Weights & Biases is the default logger. Authenticate once (`wandb login` or `WANDB_API_KEY`). For a fully local run:

```bash
python scripts/train.py logging=disabled
```

Hydra writes each run under `outputs/`. Those directories and `wandb/` are gitignored.

## Data layout

| Path | Role |
|---|---|
| `data/fixtures/sample.pgn` | Small checked-in PGN (default) |
| `data/raw/` | Heavy dumps, gitignored. Place `lichess_db_standard_rated_2013-01.pgn.zst` here |
| `data/processed/<dataset>/moves.txt` | Cached UCI games, one game per line |
| `data/processed/<dataset>/manifest.json` | Source identity + filter params used to build the cache |
| `data/tokenizer/<dataset>/tokenizer.json` | BPE tokenizer trained on that corpus |

Fixture and Lichess use **separate** processed/tokenizer directories so they cannot overwrite each other.

### Cache rules

`prepare_corpus` order:

1. If `data.hf_dataset.repo_id` is set, pull `moves.txt`, `manifest.json`, and `tokenizer.json` from that **dataset** repo.
2. Else reuse a local `moves.txt` when `manifest.json` matches `urls`, `format`, `max_games`, and `min_moves`.
3. Else download each URL in `data.urls` into `raw_path` (skipped if the file is already there), stream-parse PGN to UCI, then train a tokenizer.
4. After a rebuild, if `hf_dataset.push` and `repo_id` are set, upload the processed files so later machines skip PGN.

Add more Lichess months as extra YAML list entries. The download name is the URL basename (no `wget`).

The tokenizer is loaded from disk if `tokenizer.json` exists **and** the corpus was not just rebuilt. Otherwise it is trained on `moves.txt`.

Force a rebuild:

```bash
python scripts/prepare_data.py data.force_reprocess=true
python scripts/prepare_data.py data=lichess data.force_reprocess=true
```

### Prepare once, then train

Default (fixture, cheap):

```bash
python scripts/prepare_data.py
python scripts/train.py logging=disabled
```

January 2013 Lichess dump (downloads from YAML if missing):

```bash
python scripts/prepare_data.py data=lichess data.hf_dataset.repo_id=USER/nebium-lichess-uci
python scripts/train.py data=lichess data.hf_dataset.repo_id=USER/nebium-lichess-uci logging=disabled
```

The first prepare streams the `.zst` (not fully decompressed into RAM) and can push the UCI corpus to a single Hub **dataset**. Later runs pull that dataset. Cap size with `data.max_games=1000`.

## Architecture

```text
Chess PGN / PGN.zst
   │
   ▼
Chess tokenizer (BPE on space-separated UCI moves)
   │
   ▼
Token embedding
   │
   +── RoPE (applied to Q/K in attention)
   │
   ▼
┌───────────────────────────┐
│ Transformer block × N     │
│                           │
│ RMSNorm                   │
│       ↓                   │
│ Multi-head self-attention │
│   ├─ Q, K, V              │
│   ├─ Causal + pad mask    │
│   └─ Output projection    │
│       ↓                   │
│ Residual                  │
│       ↓                   │
│ RMSNorm                   │
│       ↓                   │
│ SwiGLU FFN                │
│       ↓                   │
│ Residual                  │
└───────────────────────────┘
   │
   ▼
Final RMSNorm → LM head → next-token logits
```

Training objective is causal cross-entropy on shifted move tokens (pad positions use ignore index `-100`).

### Evaluation

Each epoch, validation reports next-token metrics over labeled positions (`labels != -100`):

| Metric | Meaning |
|---|---|
| `val/accuracy` | Fraction of positions where `argmax` matches the gold next move |
| `val/top5_accuracy` | Gold move is among the five highest-probability tokens |
| `val/perplexity` | `exp(mean CE)`; lower is more confident next-move prediction |
| `val/loss` | Mean cross-entropy on those positions |

### Scaling Laws

To evaluate the Nebium family against theoretical Chinchilla power-law scaling predictions:

```bash
uv run python scripts/eval_scaling_laws.py \
  --small checkpoint_small.pt \
  --medium checkpoint_medium.pt \
  --large checkpoint_large.pt \
  --data-path data/fixtures/sample.pgn \
  --report paper_assets/SCALING_LAWS.md
```

| Piece | Module |
|---|---|
| Token embedding | `src/models/transformer/embedding.py` |
| RoPE | `src/models/transformer/rope.py` |
| MHSA | `src/models/transformer/attention.py` |
| SwiGLU / GELU FFN | `src/models/transformer/ffn.py` |
| RMSNorm / LayerNorm | `src/models/transformer/norm.py` |
| Block | `src/models/transformer/block.py` |
| Full model | `src/models/transformer/nebium.py` |

YAML knobs: `positional_encoding` (`rope` or `learned`), `activation` (`swiglu` or `gelu`), `norm` (`rmsnorm` or `layernorm`), `attention_type` (`standard`).

- `configs/model/nebium_stub.yaml` — 64-d, 4 heads, 1 layer (default, fixture, ~8K params)
- `configs/model/nebium_base.yaml` — 512-d, 8 heads, 6 layers (~6M params)
- `configs/model/nebium_117m.yaml` — Nebium-Small (117M params, GPT-2 Small scale)
- `configs/model/nebium_345m.yaml` — Nebium-Medium (345M params, GPT-2 Medium scale)
- `configs/model/nebium_762m.yaml` — Nebium-Large (762M params, GPT-2 Large scale)
- `configs/model/nebium_1_5b.yaml` — Nebium-XL (1.5B params)

## How to run

From the repo root, with the project environment active (`uv sync` / `uv run`):

```bash
# Fixture model + fixture data + W&B
python scripts/train.py

# Local, no W&B
python scripts/train.py logging=disabled

# Offline W&B
python scripts/train.py logging.mode=offline

# Lichess 2013 corpus (use a larger model when you mean it)
python scripts/train.py data=lichess model=nebium_base training=default logging=disabled

# Train the Nebium scaling family on Kaggle
python scripts/train.py --config-name kaggle_small
python scripts/train.py --config-name kaggle_medium
python scripts/train.py --config-name kaggle_large

# Push the final checkpoint to Hugging Face (HF_TOKEN or huggingface-cli login)
python scripts/train.py hub=huggingface hub.repo_id=USER/nebium
```

## Kaggle (end to end)

Use [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb). Turn **Internet** on. Add secrets `HF_TOKEN` and optionally `WANDB_API_KEY`.

### Quick pipeline check (smoke test)
Run `--smoke-test` to verify all pipeline stages (data ingestion, tokenizer training, model initialization, forward/backward pass, validation metrics, puzzle eval, GGUF export, and Hub export packaging) in ~5 seconds without downloading large datasets or overwriting production checkpoints:

```bash
python scripts/train.py --config-name kaggle --smoke-test \
  data.hf_dataset.repo_id=USER/nebium-lichess-uci \
  hub.repo_id=USER/nebium
```

### Full training run

```bash
python scripts/train.py --config-name kaggle \
  data.hf_dataset.repo_id=USER/nebium-lichess-uci \
  hub.repo_id=USER/nebium
```

`configs/kaggle.yaml` uses `nebium_base`, GPU batch 32, writes under `/kaggle/working`, and downloads

`https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst`

First session: download → process → push dataset → train → optional model push. Later sessions: pull the dataset → train.

Auth for the Hub is `HF_TOKEN` or `huggingface-cli login`. Tokens are never stored in the repo. The export folder (`export/`) contains `model.pt`, `model_config.json`, `tokenizer.json`, and a short model card. Default `hub=disabled` so fixture runs do not upload.

Useful overrides:

```bash
python scripts/train.py training.epochs=1 training.batch_size=4
python scripts/train.py model.n_layers=2 model.d_model=128
python scripts/train.py data.max_games=500
```

Scripts:

- `scripts/prepare_data.py` — cache corpus + tokenizer only
- `scripts/train.py` — prepare (cache-aware) then train

## Config map

```text
configs/config.yaml          # defaults: stub model, fixture training/data, wandb
configs/kaggle.yaml          # Kaggle compose: base model + Lichess URLs
configs/model/               # nebium_stub, nebium_base
configs/training/            # fixture, default, kaggle
configs/data/                # fixture, lichess, kaggle
configs/logging/             # wandb, disabled
configs/hub/                 # disabled (default), huggingface
```

## Project layout

```text
src/data/           # PGN sources, cache, dataset, tokenizer
src/models/         # Nebium transformer
src/training/       # Decoupled loop (optimizer, AMP)
src/evaluation/     # Accuracy, top-5, perplexity
src/hub/            # Hugging Face export
src/logging/        # Logger protocol, W&B, no-op
scripts/            # Hydra entrypoints
notebooks/          # Exploratory tokenizer notes
```
