import json
from pathlib import Path

import torch
from huggingface_hub import HfApi
from omegaconf import DictConfig, OmegaConf

from src.data.tokenizer import ChessTokenizer


def _model_card(cfg: DictConfig, metrics: dict[str, float]) -> str:
    lines = [
        "---",
        "library_name: pytorch",
        "tags:",
        "- chess",
        "- causal-lm",
        "- nebium",
        "---",
        "",
        "# Nebium",
        "",
        "Causal Transformer for self-supervised next-move prediction on chess games.",
        "",
        "## Architecture",
        "",
        f"- d_model: {cfg.model.d_model}",
        f"- n_heads: {cfg.model.n_heads}",
        f"- n_layers: {cfg.model.n_layers}",
        f"- vocab_size: {cfg.model.vocab_size}",
        f"- max_seq_len: {cfg.model.max_seq_len}",
        f"- positional_encoding: {cfg.model.get('positional_encoding', 'rope')}",
        f"- activation: {cfg.model.get('activation', 'swiglu')}",
        f"- norm: {cfg.model.get('norm', 'rmsnorm')}",
        "",
        "## Last validation metrics",
        "",
    ]
    for key in ("val/loss", "val/accuracy", "val/top5_accuracy", "val/perplexity"):
        if key in metrics:
            lines.append(f"- `{key}`: {metrics[key]}")
    lines.append("")
    return "\n".join(lines)


def export_checkpoint(
    model: torch.nn.Module,
    tokenizer: ChessTokenizer,
    cfg: DictConfig,
    metrics: dict[str, float],
    export_dir: Path,
) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), export_dir / "model.pt")
    config = OmegaConf.to_container(cfg.model, resolve=True)
    (export_dir / "model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tokenizer.save(str(export_dir / "tokenizer.json"))
    (export_dir / "README.md").write_text(_model_card(cfg, metrics), encoding="utf-8")
    return export_dir


def push_to_hub(
    model: torch.nn.Module,
    tokenizer: ChessTokenizer,
    cfg: DictConfig,
    metrics: dict[str, float],
    export_dir: Path | None = None,
) -> str | None:
    if not bool(cfg.hub.get("push", False)):
        return None
    repo_id = cfg.hub.repo_id
    if not repo_id:
        raise ValueError("hub.repo_id is required when hub.push=true (e.g. USER/nebium).")
    dest = Path(export_dir) if export_dir is not None else Path("export")
    export_checkpoint(model, tokenizer, cfg, metrics, dest)
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=bool(cfg.hub.get("private", True)), exist_ok=True)
    api.upload_folder(
        folder_path=str(dest),
        repo_id=repo_id,
        commit_message=str(cfg.hub.get("commit_message", "Add Nebium checkpoint")),
    )
    return repo_id
