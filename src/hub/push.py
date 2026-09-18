"""
Hub push utilities: export checkpoints and generate publication-standard model cards
for both PyTorch weights and dedicated GGUF repositories across Nebium-Small,
Nebium-Medium, and Nebium-Large.
"""

import json
import math
import os
import shutil
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
            "Nebium-Small is a 117-million-parameter causal Transformer trained for autoregressive "
            "next-chess-move prediction over Lichess UCI move sequences."
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
            "Nebium-Medium is a 345-million-parameter causal Transformer balancing sequence "
            "quality and inference throughput for production evaluation."
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
            "in the Nebium family, optimized for move legality and tactical sequence reasoning."
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


def _detect_family_key(cfg: DictConfig) -> str:
    """Infer which family member this config corresponds to (by d_model / n_layers)."""
    if not hasattr(cfg, "model"):
        return "nebium_117m"
    d = int(cfg.model.get("d_model", 768))
    n = int(cfg.model.get("n_layers", 12))
    for key, meta in _FAMILY.items():
        if meta["d_model"] == d and meta["n_layers"] == n:
            return key
    return "nebium_117m"


def _chinchilla_loss_estimate(n_params: int, n_tokens: int) -> float:
    """
    Hoffmann et al. (2022) Chinchilla power-law loss estimate:
        L(N, D) = E + A/N^alpha + B/D^beta
    where E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28.
    """
    E, A, alpha, B, beta = 1.69, 406.4, 0.34, 410.7, 0.28
    return E + A / (n_params ** alpha) + B / (n_tokens ** beta)


def _model_card(
    cfg: DictConfig,
    metrics: dict[str, float],
    model: torch.nn.Module | None = None,
    repo_id: str | None = None,
    gguf_repo_id: str | None = None,
) -> str:
    """Generate a rich, per-model Hugging Face model card with scaling law section."""
    family_key = _detect_family_key(cfg)
    meta = _FAMILY.get(family_key, _FAMILY["nebium_117m"])
    model_name = meta["name"]
    params_label = meta["params_label"]
    tier = meta["tier"]
    description = meta["description"]
    chinchilla_tokens_b = meta["chinchilla_tokens_b"]

    actual_params = _count_params(model) if model is not None else None
    actual_params_str = f"{actual_params / 1e6:.1f}M" if actual_params else params_label

    # --- YAML front-matter ---
    family_tags = [f"nebium-{tier}"]
    yaml_tags = ["chess", "causal-lm", "nebium", "transformer", "rope", "swiglu", "rmsnorm"] + family_tags
    tag_lines = "\n".join(f"- {t}" for t in yaml_tags)

    hf_metrics_yaml = ""
    for k, hf_k in [
        ("val/loss", "val_loss"),
        ("val/accuracy", "accuracy"),
        ("val/perplexity", "perplexity"),
        ("val/legal_move_rate", "legal_move_rate"),
        ("val/puzzle_accuracy", "puzzle_accuracy"),
    ]:
        if k in metrics:
            hf_metrics_yaml += f"      - type: {hf_k}\n        value: {float(metrics[k]):.6f}\n"

    model_index_block = ""
    if hf_metrics_yaml.strip():
        model_index_block = f"""model-index:
- name: {model_name}
  results:
  - task:
      type: text-generation
      name: Chess Next Move Prediction
    dataset:
      name: Lichess UCI Move Sequences
      type: nabin2004/nebium-lichess-uci
    metrics:
{hf_metrics_yaml.rstrip()}"""

    front_matter = f"""---
language:
- en
license: mit
library_name: pytorch
tags:
{tag_lines}
datasets:
- nabin2004/nebium-lichess-uci
pipeline_tag: text-generation
{model_index_block}
---"""

    val_loss = metrics.get("val/loss", float("nan"))
    val_acc = metrics.get("val/accuracy", float("nan"))
    val_top5 = metrics.get("val/top5_accuracy", float("nan"))
    val_ppl = metrics.get("val/perplexity", float("nan"))
    legal_rate = metrics.get("val/legal_move_rate", float("nan"))
    puzzle_acc = metrics.get("val/puzzle_accuracy", float("nan"))

    scaling_section = ""
    if chinchilla_tokens_b is not None:
        n_params_est = actual_params if actual_params else int(meta["d_model"] ** 2 * meta["n_layers"] * 12)
        d_optimal = int(chinchilla_tokens_b * 1e9)
        d_trained = int(cfg.data.get("max_games", 245293)) * 50
        L_optimal = _chinchilla_loss_estimate(n_params_est, d_optimal)
        L_trained = _chinchilla_loss_estimate(n_params_est, max(d_trained, 1_000_000))
        scaling_section = f"""
## Scaling Law Analysis (Hoffmann et al. 2022)

Chinchilla power-law formulation:

$$L(N, D) = 1.69 + \\frac{{406.4}}{{N^{{0.34}}}} + \\frac{{410.7}}{{D^{{0.28}}}}$$

| Parameter / Metric | Value |
|---|---|
| Model Parameters ($N$) | {actual_params_str} |
| Chinchilla-Optimal Token Budget ($D^*$) | ~{chinchilla_tokens_b}B tokens |
| Compute-Optimal Expected Loss ($L_{{optimal}}$) | {L_optimal:.4f} nats |
| Approximate Trained Tokens ($D$) | ~{d_trained / 1e6:.1f}M tokens |
| Theoretical Loss at Current Tokens | {L_trained:.4f} nats |
| Empirical Validation Loss | {val_loss:.4f} nats |
"""

    gguf_link = f"- **GGUF Repository**: [{gguf_repo_id}](https://huggingface.co/{gguf_repo_id})" if gguf_repo_id else ""
    pytorch_link = f"- **PyTorch Repository**: [{repo_id}](https://huggingface.co/{repo_id})" if repo_id else ""

    body = f"""
# {model_name} ({actual_params_str})

{description}

{pytorch_link}
{gguf_link}
- **Source Repository**: [github.com/nabin2004/nebium](https://github.com/nabin2004/nebium)

## Nebium Model Family Architecture Overview

{_FAMILY_TABLE}

Architectural Primitives:
- **Rotary Position Embeddings (RoPE)** on attention query and key projections ($\\theta = 10000$)
- **SwiGLU** feed-forward transformation
- **RMSNorm** pre-normalization
- **Causal mask** with padding token masking
- **Byte-Pair Encoding (BPE)** tokenizer trained on UCI move plies

## Architectural Specifications

| Hyperparameter | Value |
|---|---|
| Model Tier | **{model_name}** |
| Parameter Count | **{actual_params_str}** |
| Hidden Dimension ($d_{{model}}$) | {cfg.model.d_model} |
| Attention Heads ($n_{{heads}}$) | {cfg.model.n_heads} |
| Transformer Layers ($n_{{layers}}$) | {cfg.model.n_layers} |
| Max Context Length ($L_{{max}}$) | {cfg.model.max_seq_len} |
| Vocabulary Size ($V$) | {cfg.model.vocab_size} |
| Positional Embedding | {cfg.model.get('positional_encoding', 'rope')} |
| Activation Function | {cfg.model.get('activation', 'swiglu')} |
| Layer Normalization | {cfg.model.get('norm', 'rmsnorm')} |

## Validation & Benchmark Results

| Metric | Measured Value |
|---|---|
| Validation Loss | {val_loss:.6f} |
| Validation Perplexity | {val_ppl:.4f} |
| Next-Token Top-1 Accuracy | {val_acc * 100:.2f}% |
| Next-Token Top-5 Accuracy | {val_top5 * 100:.2f}% |
| Empirical Move Legality Rate | {legal_rate * 100:.2f}% |
| Tactical Puzzle Accuracy | {puzzle_acc * 100:.2f}% |
{scaling_section}

## Python Usage Example

```python
import json
import torch
from src.models.transformer.nebium import Nebium
from src.data.tokenizer import ChessTokenizer

tokenizer = ChessTokenizer()
tokenizer.load("tokenizer.json")

with open("model_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

model = Nebium(**config)
state_dict = torch.load("model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

prompt = "e2e4 e7e5 g1f3"
input_ids = torch.tensor([[tokenizer.bos_id] + tokenizer.encode(prompt)], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    output = model.generate(input_ids, attention_mask, max_new_tokens=10, temperature=0.7)

print("Continuation:", tokenizer.decode(output[0].tolist()))
```

## License

MIT License.
"""
    return front_matter + "\n" + body


def _gguf_model_card(
    cfg: DictConfig,
    metrics: dict[str, float],
    repo_id: str | None = None,
    gguf_repo_id: str | None = None,
) -> str:
    """Generate a dedicated Hugging Face model card for the GGUF repository."""
    family_key = _detect_family_key(cfg)
    meta = _FAMILY.get(family_key, _FAMILY["nebium_117m"])
    model_name = meta["name"]
    tier = meta["tier"]
    params_label = meta["params_label"]

    return f"""---
language:
- en
license: mit
tags:
- chess
- causal-lm
- gguf
- llama.cpp
- ollama
- nebium
- nebium-{tier}
pipeline_tag: text-generation
---

# {model_name}-GGUF

Quantized and FP16 GGUF format binaries for **{model_name}** ({params_label} parameters).

Designed for low-latency CPU and GPU execution with [llama.cpp](https://github.com/ggerganov/llama.cpp) and [Ollama](https://ollama.ai).

- **PyTorch Base Model**: [{repo_id}](https://huggingface.co/{repo_id})
- **GGUF Repository**: [{gguf_repo_id}](https://huggingface.co/{gguf_repo_id})
- **Source Repository**: [github.com/nabin2004/nebium](https://github.com/nabin2004/nebium)

---

## Artifacts

| Filename | Precision | Description |
|---|---|---|
| `nebium-{tier}.gguf` | FP16 | Full-precision baseline export |
| `tokenizer.json` | Tokenizer | BPE vocabulary and merge definitions |

---

## Inference with llama.cpp

```bash
# Clone and compile llama.cpp
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make

# Download GGUF binary
huggingface-cli download {gguf_repo_id} nebium-{tier}.gguf --local-dir .

# Run prompt continuation
./llama-cli -m nebium-{tier}.gguf -p "e2e4 e7e5 g1f3" -n 25 --temp 0.7
```

---

## Inference with Ollama

```dockerfile
# Modelfile
FROM ./nebium-{tier}.gguf
PARAMETER temperature 0.7
PARAMETER stop "<|eos|>"
SYSTEM You are an autoregressive chess next-move prediction model using UCI move notation.
```

```bash
ollama create nebium-{tier} -f Modelfile
ollama run nebium-{tier} "e2e4 e7e5"
```

## License

MIT License.
"""


# ---------------------------------------------------------------------------
# Export & Push
# ---------------------------------------------------------------------------

def export_checkpoint(
    model: torch.nn.Module,
    tokenizer: ChessTokenizer,
    cfg: DictConfig,
    metrics: dict[str, float],
    export_dir: Path,
    repo_id: str | None = None,
    gguf_repo_id: str | None = None,
) -> dict[str, Path]:
    """
    Exports model weights, configurations, tokenizer, and GGUF binary to staging directories.
    Produces both base PyTorch directory and dedicated GGUF directory.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    pytorch_dir = export_dir / "pytorch"
    gguf_dir = export_dir / "gguf"
    pytorch_dir.mkdir(parents=True, exist_ok=True)
    gguf_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    # 1. PyTorch directory
    torch.save(state_dict, pytorch_dir / "model.pt")
    config = OmegaConf.to_container(cfg.model, resolve=True)
    (pytorch_dir / "model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tokenizer.save(str(pytorch_dir / "tokenizer.json"))
    (pytorch_dir / "README.md").write_text(
        _model_card(cfg, metrics, model=model, repo_id=repo_id, gguf_repo_id=gguf_repo_id),
        encoding="utf-8",
    )

    # Backwards compatibility flat files in export_dir root
    torch.save(state_dict, export_dir / "model.pt")
    (export_dir / "model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tokenizer.save(str(export_dir / "tokenizer.json"))
    (export_dir / "README.md").write_text(
        _model_card(cfg, metrics, model=model, repo_id=repo_id, gguf_repo_id=gguf_repo_id),
        encoding="utf-8",
    )

    # 2. GGUF directory
    family_key = _detect_family_key(cfg)
    tier = _FAMILY.get(family_key, {}).get("tier", "small")
    gguf_filename = f"nebium-{tier}.gguf"

    has_gguf = False
    if bool(cfg.hub.get("export_gguf", True)):
        try:
            gguf_path = gguf_dir / gguf_filename
            export_to_gguf(model, tokenizer, gguf_path, precision=str(cfg.hub.get("gguf_precision", "fp16")))
            tokenizer.save(str(gguf_dir / "tokenizer.json"))
            (gguf_dir / "README.md").write_text(
                _gguf_model_card(cfg, metrics, repo_id=repo_id, gguf_repo_id=gguf_repo_id),
                encoding="utf-8",
            )
            # Also keep a copy at export_dir / "nebium.gguf"
            shutil.copy2(gguf_path, export_dir / "nebium.gguf")
            has_gguf = True
        except Exception as exc:
            print(f"[Hub Export] Warning: Failed to export GGUF: {exc}")

    return {"pytorch": pytorch_dir, "gguf": gguf_dir}


def push_to_hub(
    model: torch.nn.Module,
    tokenizer: ChessTokenizer,
    cfg: DictConfig,
    metrics: dict[str, float],
    export_dir: Path | None = None,
) -> dict[str, str] | None:
    """
    Exports model artifacts and uploads to both the base PyTorch model repo
    and the dedicated companion GGUF repo on the Hugging Face Hub.
    """
    if not bool(cfg.hub.get("push", False)):
        return None

    api = HfApi()
    try:
        user_info = api.whoami()
        username = user_info.get("name", "nabin2004")
    except Exception:
        username = "nabin2004"

    family_key = _detect_family_key(cfg)
    tier = _FAMILY.get(family_key, {}).get("tier", "small")

    # Resolve target repository IDs
    base_repo_id = cfg.hub.get("repo_id") or f"{username}/nebium-{tier}"
    gguf_repo_id = cfg.hub.get("gguf_repo_id") or f"{username}/nebium-{tier}-gguf"
    is_private = bool(cfg.hub.get("private", False))

    dest = Path(export_dir) if export_dir is not None else Path("export")
    exported_dirs = export_checkpoint(
        model,
        tokenizer,
        cfg,
        metrics,
        dest,
        repo_id=base_repo_id,
        gguf_repo_id=gguf_repo_id,
    )

    pushed = {}

    # 1. Push PyTorch Base Model Repository
    print(f"[Hub Push] Uploading PyTorch checkpoint to https://huggingface.co/{base_repo_id}...")
    try:
        api.create_repo(repo_id=base_repo_id, private=is_private, exist_ok=True)
        api.upload_folder(
            folder_path=str(exported_dirs["pytorch"]),
            repo_id=base_repo_id,
            commit_message=str(cfg.hub.get("commit_message", f"Update {tier} weights and metrics")),
        )
        pushed["model"] = base_repo_id
        print(f"[Hub Push] Successfully pushed base model: https://huggingface.co/{base_repo_id}")
    except Exception as exc:
        print(f"[Hub Push] Error uploading base model to {base_repo_id}: {exc}")

    # 2. Push Dedicated GGUF Repository
    if bool(cfg.hub.get("push_gguf_repo", True)) and (exported_dirs["gguf"] / f"nebium-{tier}.gguf").exists():
        print(f"[Hub Push] Uploading GGUF binary to https://huggingface.co/{gguf_repo_id}...")
        try:
            api.create_repo(repo_id=gguf_repo_id, private=is_private, exist_ok=True)
            api.upload_folder(
                folder_path=str(exported_dirs["gguf"]),
                repo_id=gguf_repo_id,
                commit_message=str(cfg.hub.get("commit_message", f"Update {tier} GGUF binaries")),
            )
            pushed["gguf"] = gguf_repo_id
            print(f"[Hub Push] Successfully pushed GGUF model: https://huggingface.co/{gguf_repo_id}")
        except Exception as exc:
            print(f"[Hub Push] Error uploading GGUF model to {gguf_repo_id}: {exc}")

    return pushed if pushed else None
