# Nebium

A from-scratch causal Transformer for self-supervised next-move prediction on chess games. Configuration is Hydra YAML; the training loop talks to pluggable loggers and datasets, not to PGN or W&B directly.

## Design principles

1. Configuration over code
2. Pluggable components
3. Dataset-agnostic interface
4. Training loop decoupling
5. Extensibility for future models

## Setup

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

- `configs/model/nebium_stub.yaml` — 64-d, 4 heads, 1 layer (default, fixture)
- `configs/model/nebium_base.yaml` — 512-d, 8 heads, 6 layers

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

# Push the final checkpoint to Hugging Face (HF_TOKEN or huggingface-cli login)
python scripts/train.py hub=huggingface hub.repo_id=USER/nebium
```

## Kaggle (end to end)

Use [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb). Turn **Internet** on. Add secrets `HF_TOKEN` and optionally `WANDB_API_KEY`.

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
