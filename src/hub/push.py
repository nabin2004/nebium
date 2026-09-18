"""Hub push utilities: export checkpoints and generate rich per-model model cards."""

import json
import math
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from omegaconf import DictConfig, OmegaConf

from src.data.tokenizer import ChessTokenizer
from src.export.gguf_export import export_to_gguf


# ---------------------------------------------------------------------------
# Model family metadata
# ---------------------------------------------------------------------------

_FAMILY = {
    "nebium_117m": {
        "name": "Nebium-Small",
        "params_label": "117M",
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "chinchilla_tokens_b": 2.3,
        "tier": "small",
        "description": (
            "Nebium-Small is a 117-million-parameter causal Transformer for self-supervised "
            "next-chess-move prediction, equivalent in scale to GPT-2 Small. "
            "It is the most practical model for resource-constrained inference."
        ),
    },
    "nebium_345m": {
        "name": "Nebium-Medium",
        "params_label": "345M",
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "chinchilla_tokens_b": 6.9,
        "tier": "medium",
        "description": (
            "Nebium-Medium is a 345-million-parameter causal Transformer, equivalent in scale "
            "to GPT-2 Medium. It balances quality and inference speed for production deployment."
        ),
    },
    "nebium_762m": {
        "name": "Nebium-Large",
        "params_label": "762M",
        "d_model": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "chinchilla_tokens_b": 15.2,
        "tier": "large",
        "description": (
            "Nebium-Large is a 762-million-parameter causal Transformer, the flagship model "
            "in the Nebium family. It achieves the strongest tactical reasoning and legal-move "
            "generation accuracy among the three tiers."
        ),
    },
}

_FAMILY_TABLE = """\
| Model | Params | d_model | Heads | Layers | max_seq_len | Chinchilla-optimal tokens |
|---|---|---|---|---|---|---|
| Nebium-Small  | 117M | 768  | 12 | 12 | 1024 | ~2.3B |
| Nebium-Medium | 345M | 1024 | 16 | 24 | 1024 | ~6.9B |
| Nebium-Large  | 762M | 1280 | 20 | 36 | 1024 | ~15.2B |
"""


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _detect_family_key(cfg: DictConfig) -> str | None:
    """Infer which family member this config corresponds to (by d_model / n_layers)."""
    d = int(cfg.model.d_model)
    n = int(cfg.model.n_layers)
    for key, meta in _FAMILY.items():
        if meta["d_model"] == d and meta["n_layers"] == n:
            return key
    return None


def _chinchilla_loss_estimate(n_params: int, n_tokens: int) -> float:
    """
    Hoffmann et al. (2022) Chinchilla power-law loss estimate:
        L(N, D) = E + A/N^alpha + B/D^beta
    where E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28.
    Returns estimated cross-entropy loss (nats).
    """
    E, A, alpha, B, beta = 1.69, 406.4, 0.34, 410.7, 0.28
    return E + A / (n_params ** alpha) + B / (n_tokens ** beta)


def _model_card(
    cfg: DictConfig,
    metrics: dict[str, float],
    model: torch.nn.Module | None = None,
    has_gguf: bool = False,
) -> str:
    """Generate a rich, per-model Hugging Face model card with scaling law section."""
    family_key = _detect_family_key(cfg)
    meta = _FAMILY.get(family_key or "", {})
    model_name = meta.get("name", "Nebium")
    params_label = meta.get("params_label", "")
    tier = meta.get("tier", "")
    description = meta.get("description", "Causal Transformer for next-move prediction on chess games.")
    chinchilla_tokens_b = meta.get("chinchilla_tokens_b", None)

    actual_params = _count_params(model) if model is not None else None
    actual_params_str = f"{actual_params / 1e6:.1f}M" if actual_params else params_label

    # --- YAML front-matter ---
    family_tags = ["nebium-small"] if tier == "small" else (["nebium-medium"] if tier == "medium" else ["nebium-large"])
    yaml_tags = ["chess", "causal-lm", "nebium", "transformer", "gguf"] + family_tags
    tag_lines = "\n".join(f"- {t}" for t in yaml_tags)

    hf_metrics_yaml = ""
    for k, hf_k in [
        ("val/loss", "val_loss"),
        ("val/accuracy", "val_accuracy"),
        ("val/perplexity", "val_perplexity"),
        ("val/legal_move_rate", "legal_move_rate"),
    ]:
        if k in metrics:
            hf_metrics_yaml += f"  - type: {hf_k}\n    value: {metrics[k]:.6f}\n"

    front_matter = f"""---
library_name: pytorch
language:
- en
tags:
{tag_lines}
model-index:
- name: {model_name}
  results:
  - task:
      type: chess-next-move-prediction
    metrics:
{hf_metrics_yaml or '    []'}
---"""

    # --- Body ---
    val_loss = metrics.get("val/loss", float("nan"))
    val_acc = metrics.get("val/accuracy", float("nan"))
    val_top5 = metrics.get("val/top5_accuracy", float("nan"))
    val_ppl = metrics.get("val/perplexity", float("nan"))
    legal_rate = metrics.get("val/legal_move_rate", float("nan"))
    puzzle_acc = metrics.get("val/puzzle_accuracy", float("nan"))

    # Scaling law section
    scaling_section = ""
    if chinchilla_tokens_b is not None:
        n_params_est = actual_params if actual_params else int(meta.get("d_model", 512) ** 2 * meta.get("n_layers", 6) * 12)
        d_optimal = int(chinchilla_tokens_b * 1e9)
        d_trained = int(cfg.data.get("max_games", 245293)) * 50  # rough tokens estimate (avg 50 moves/game * avg tokens)
        L_optimal = _chinchilla_loss_estimate(n_params_est, d_optimal)
        L_trained = _chinchilla_loss_estimate(n_params_est, max(d_trained, 1_000_000))
        scaling_section = f"""
## Scaling Law Analysis (Chinchilla, Hoffmann et al. 2022)

The Chinchilla power-law loss estimate for this model:

```
L(N, D) = E + A/N^alpha + B/D^beta
where E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28
```

| Quantity | Value |
|---|---|
| Model parameters (N) | {actual_params_str} |
| Chinchilla-optimal token budget | ~{chinchilla_tokens_b}B tokens |
| Estimated optimal loss at {chinchilla_tokens_b}B tokens | {L_optimal:.4f} nats |
| Approximate tokens trained | ~{d_trained / 1e6:.1f}M tokens |
| Estimated loss at training tokens | {L_trained:.4f} nats |
| Observed validation loss | {val_loss:.4f} nats |

> **Note**: Chinchilla estimates assume a generic autoregressive LM on web text. Chess UCI token
> distributions are more structured, so actual loss may differ. The table is provided for
> **comparative scaling-law analysis** across the three Nebium tiers.
"""

    body = f"""
# {model_name} ({actual_params_str})

{description}

This model is part of the **Nebium family** — a trio of causal Transformers trained on
self-supervised next-move prediction over Lichess UCI game sequences.

## Nebium Model Family

{_FAMILY_TABLE}

All three models share the same architecture primitives:
- **RoPE** (Rotary Position Embeddings) on Q/K in every attention head
- **SwiGLU** feed-forward network
- **RMSNorm** pre-normalization
- **Causal + padding mask** for variable-length game sequences
- **BPE chess tokenizer** trained on UCI move sequences

## Architecture — {model_name}

| Hyperparameter | Value |
|---|---|
| Parameters | {actual_params_str} |
| d_model | {cfg.model.d_model} |
| n_heads | {cfg.model.n_heads} |
| n_layers | {cfg.model.n_layers} |
| max_seq_len | {cfg.model.max_seq_len} |
| vocab_size | {cfg.model.vocab_size} |
| positional_encoding | {cfg.model.get('positional_encoding', 'rope')} |
| activation | {cfg.model.get('activation', 'swiglu')} |
| norm | {cfg.model.get('norm', 'rmsnorm')} |

## Validation Metrics

| Metric | Value |
|---|---|
| Validation Loss | {val_loss:.6f} |
| Validation Perplexity | {val_ppl:.4f} |
| Top-1 Accuracy | {val_acc * 100:.2f}% |
| Top-5 Accuracy | {val_top5 * 100:.2f}% |
| Legal Move Rate | {legal_rate * 100:.2f}% |
| Puzzle Accuracy | {puzzle_acc * 100:.2f}% |
{scaling_section}

## Artifacts

| File | Description |
|---|---|
| `model.pt` | PyTorch state dict (raw weights) |
| `model_config.json` | Architecture configuration JSON |
| `tokenizer.json` | BPE chess tokenizer |
{("| `nebium.gguf` | GGUF format for local inference via llama.cpp / Ollama |" if has_gguf else "")}

## Usage

```python
import torch
from omegaconf import OmegaConf
from src.models.transformer.nebium import Nebium
from src.data.tokenizer import ChessTokenizer

# Load tokenizer
tokenizer = ChessTokenizer()
tokenizer.load("tokenizer.json")

# Load model
config = OmegaConf.load("model_config.json")
model = Nebium(**config)
state_dict = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Generate moves from a starting position
prompt = "e2e4 e7e5"
input_ids = torch.tensor([[tokenizer.bos_id] + tokenizer.encode(prompt)])
mask = torch.ones_like(input_ids)

with torch.no_grad():
    output = model.generate(input_ids, mask, max_new_tokens=10, temperature=0.7)
moves = tokenizer.decode(output[0].tolist())
print("Generated game:", moves)
```

## Training

Trained with the [Nebium](https://github.com/nabin2004/nebium) framework using:

```bash
python scripts/train.py --config-name kaggle_{tier} \\
  data.hf_dataset.repo_id=USER/nebium-lichess-uci \\
  hub.repo_id=USER/nebium-{tier}
```

## License

MIT
"""

    return front_matter + body


# ---------------------------------------------------------------------------
# Export & Push
# ---------------------------------------------------------------------------

def export_checkpoint(
    model: torch.nn.Module,
    tokenizer: ChessTokenizer,
    cfg: DictConfig,
    metrics: dict[str, float],
    export_dir: Path,
) -> Path:
    """
    Exports PyTorch model weights, JSON configuration, tokenizer, GGUF binary,
    and a generated Markdown model card to an export directory.

    Args:
        model: Trained PyTorch model.
        tokenizer: Initialized ChessTokenizer.
        cfg: Hydra configuration dictionary.
        metrics: Dictionary of evaluation metrics.
        export_dir: Output directory path.

    Returns:
        Path to the populated export directory.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state_dict, export_dir / "model.pt")
    config = OmegaConf.to_container(cfg.model, resolve=True)
    (export_dir / "model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tokenizer.save(str(export_dir / "tokenizer.json"))

    has_gguf = False
    if bool(cfg.hub.get("export_gguf", True)):
        try:
            gguf_path = export_dir / "nebium.gguf"
            export_to_gguf(model, tokenizer, gguf_path, precision=str(cfg.hub.get("gguf_precision", "fp16")))
            has_gguf = True
        except Exception as exc:
            print(f"[Hub Export] Warning: Failed to export GGUF: {exc}")

    (export_dir / "README.md").write_text(
        _model_card(cfg, metrics, model=model, has_gguf=has_gguf),
        encoding="utf-8",
    )
    return export_dir


def push_to_hub(
    model: torch.nn.Module,
    tokenizer: ChessTokenizer,
    cfg: DictConfig,
    metrics: dict[str, float],
    export_dir: Path | None = None,
) -> str | None:
    """
    Exports model artifacts and uploads the directory to the Hugging Face Hub.

    Args:
        model: Trained PyTorch model.
        tokenizer: Initialized ChessTokenizer.
        cfg: Hydra configuration dictionary.
        metrics: Evaluation metrics dictionary.
        export_dir: Optional staging directory (default: 'export').

    Returns:
        Hugging Face repository ID string if pushed, or None if push disabled.

    Raises:
        ValueError: If `hub.repo_id` is not configured when push is enabled.
    """
    if not bool(cfg.hub.get("push", False)):
        return None
    repo_id = cfg.hub.repo_id
    if not repo_id:
        raise ValueError("hub.repo_id is required when hub.push=true (e.g. USER/nebium-small).")
    dest = Path(export_dir) if export_dir is not None else Path("export")
    export_checkpoint(model, tokenizer, cfg, metrics, dest)
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=bool(cfg.hub.get("private", True)), exist_ok=True)
    api.upload_folder(
        folder_path=str(dest),
        repo_id=repo_id,
        commit_message=str(cfg.hub.get("commit_message", "Add Nebium checkpoint and GGUF model")),
    )
    return repo_id

