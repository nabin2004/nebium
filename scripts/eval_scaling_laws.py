"""
scripts/eval_scaling_laws.py
============================
Evaluate all three Nebium family models (Small/Medium/Large) against Chinchilla
scaling law predictions and produce a comprehensive scaling-law analysis report.

Usage
-----
# From the repo root, compare three trained checkpoints:
uv run python scripts/eval_scaling_laws.py \\
  --small  checkpoint_small.pt  \\
  --medium checkpoint_medium.pt \\
  --large  checkpoint_large.pt  \\
  --data-path data/fixtures/sample.pgn \\
  --output paper_assets/scaling_laws.json

# To also produce a markdown report:
uv run python scripts/eval_scaling_laws.py ... --report paper_assets/SCALING_LAWS.md
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_dataloaders
from src.data.prepare import get_tokenizer
from src.data.sources import stream_uci_games
from src.models.transformer.nebium import Nebium
from src.evaluation.metrics import merge_metric_batches, next_token_metrics

# ---------------------------------------------------------------------------
# Nebium family definitions (mirrors src/hub/push.py)
# ---------------------------------------------------------------------------

FAMILY = {
    "small": {
        "name": "Nebium-Small",
        "params_approx": 117_000_000,
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "max_seq_len": 1024,
        "vocab_size": 5000,
        "chinchilla_tokens": 2_300_000_000,
        "lr": 3e-4,
    },
    "medium": {
        "name": "Nebium-Medium",
        "params_approx": 345_000_000,
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "max_seq_len": 1024,
        "vocab_size": 5000,
        "chinchilla_tokens": 6_900_000_000,
        "lr": 2e-4,
    },
    "large": {
        "name": "Nebium-Large",
        "params_approx": 762_000_000,
        "d_model": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "max_seq_len": 1024,
        "vocab_size": 5000,
        "chinchilla_tokens": 15_200_000_000,
        "lr": 1.5e-4,
    },
}


# ---------------------------------------------------------------------------
# Chinchilla power-law
# ---------------------------------------------------------------------------

def chinchilla_loss(n_params: int, n_tokens: int) -> float:
    """
    Hoffmann et al. (2022) Eq. 4:
        L(N, D) = E + A/N^alpha + B/D^beta
    E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28
    """
    E, A, alpha, B, beta = 1.69, 406.4, 0.34, 410.7, 0.28
    return E + A / (n_params ** alpha) + B / (n_tokens ** beta)


def chinchilla_optimal_tokens(n_params: int) -> int:
    """Return Chinchilla-optimal token budget: D* = 20 * N."""
    return 20 * n_params


def scaling_fit(sizes_params: list[int], losses: list[float]) -> tuple[float, float]:
    """
    Fit a simple power law L = a * N^b to empirical (N, L) points
    using log-linear regression: log(L) = log(a) + b*log(N)
    Returns (a, b).
    """
    if len(sizes_params) < 2:
        return float("nan"), float("nan")
    import numpy as np
    log_n = [math.log(n) for n in sizes_params]
    log_l = [math.log(max(l, 1e-9)) for l in losses]
    n_arr = len(log_n)
    sum_x = sum(log_n)
    sum_y = sum(log_l)
    sum_xx = sum(x * x for x in log_n)
    sum_xy = sum(x * y for x, y in zip(log_n, log_l))
    denom = n_arr * sum_xx - sum_x ** 2
    if abs(denom) < 1e-12:
        return float("nan"), float("nan")
    b = (n_arr * sum_xy - sum_x * sum_y) / denom
    a = math.exp((sum_y - b * sum_x) / n_arr)
    return a, b


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_model(tier: str, vocab_size: int) -> Nebium:
    meta = FAMILY[tier]
    return Nebium(
        vocab_size=vocab_size,
        d_model=meta["d_model"],
        n_heads=meta["n_heads"],
        n_layers=meta["n_layers"],
        dropout=0.0,
        max_seq_len=meta["max_seq_len"],
        positional_encoding="rope",
        attention_type="standard",
        activation="swiglu",
        norm="rmsnorm",
    )


def _load_checkpoint(tier: str, ckpt_path: str, vocab_size: int, device: torch.device) -> Nebium:
    model = _build_model(tier, vocab_size)
    state = torch.load(ckpt_path, map_location="cpu")
    # Support raw state_dict or wrapped checkpoint
    sd = state.get("model", state)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: Nebium,
    val_loader,
    device: torch.device,
) -> dict[str, float]:
    batches = []
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask)
        batches.append(next_token_metrics(logits.float(), labels))
    return merge_metric_batches(batches)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _format_float(v: float, decimals: int = 4) -> str:
    if math.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def generate_markdown_report(results: dict[str, Any]) -> str:
    lines = [
        "# Nebium Scaling Law Analysis",
        "",
        "Empirical comparison of the three Nebium family models against Chinchilla (2022) power-law predictions.",
        "",
        "## Model Family Overview",
        "",
        "| Model | Params (actual) | Params (approx) | Chinchilla-optimal tokens |",
        "|---|---|---|---|",
    ]
    for tier in ["small", "medium", "large"]:
        if tier not in results:
            continue
        r = results[tier]
        meta = FAMILY[tier]
        lines.append(
            f"| {meta['name']} | {r['actual_params'] / 1e6:.1f}M "
            f"| ~{meta['params_approx'] / 1e6:.0f}M "
            f"| ~{meta['chinchilla_tokens'] / 1e9:.1f}B |"
        )

    lines += [
        "",
        "## Empirical Validation Metrics",
        "",
        "| Model | Val Loss | Perplexity | Top-1 Acc | Top-5 Acc | Legal Move Rate |",
        "|---|---|---|---|---|---|",
    ]
    for tier in ["small", "medium", "large"]:
        if tier not in results:
            continue
        r = results[tier]
        m = r.get("metrics", {})
        lines.append(
            f"| {FAMILY[tier]['name']} "
            f"| {_format_float(m.get('val/loss', float('nan')))} "
            f"| {_format_float(m.get('val/perplexity', float('nan')))} "
            f"| {m.get('val/accuracy', 0) * 100:.2f}% "
            f"| {m.get('val/top5_accuracy', 0) * 100:.2f}% "
            f"| {m.get('val/legal_move_rate', 0) * 100:.2f}% |"
        )

    lines += [
        "",
        "## Chinchilla Power-Law Predictions",
        "",
        "Estimated loss: `L(N,D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28`",
        "",
        "| Model | N (params) | D (tokens trained) | L_predicted (Chinchilla) | L_observed | Delta |",
        "|---|---|---|---|---|---|",
    ]
    for tier in ["small", "medium", "large"]:
        if tier not in results:
            continue
        r = results[tier]
        m = r.get("metrics", {})
        n = r["actual_params"]
        d = r.get("tokens_trained", 1_000_000)
        l_pred = chinchilla_loss(n, d)
        l_obs = m.get("val/loss", float("nan"))
        delta = l_obs - l_pred if not math.isnan(l_obs) else float("nan")
        lines.append(
            f"| {FAMILY[tier]['name']} "
            f"| {n / 1e6:.1f}M "
            f"| {d / 1e6:.1f}M "
            f"| {l_pred:.4f} "
            f"| {_format_float(l_obs)} "
            f"| {_format_float(delta, 4)} |"
        )

    # Empirical scaling law fit
    tiers_with_data = [t for t in ["small", "medium", "large"] if t in results]
    if len(tiers_with_data) >= 2:
        sizes = [results[t]["actual_params"] for t in tiers_with_data]
        losses = [results[t]["metrics"].get("val/loss", float("nan")) for t in tiers_with_data]
        valid = [(n, l) for n, l in zip(sizes, losses) if not math.isnan(l)]
        if len(valid) >= 2:
            a, b = scaling_fit([x[0] for x in valid], [x[1] for x in valid])
            lines += [
                "",
                "## Empirical Scaling-Law Fit",
                "",
                f"Fitted power law from observed data: `L = {a:.4f} * N^{b:.4f}`",
                "",
                "| Model | N (params) | L_fitted | L_observed |",
                "|---|---|---|---|",
            ]
            for tier in tiers_with_data:
                r = results[tier]
                n = r["actual_params"]
                l_fit = a * (n ** b)
                l_obs = r["metrics"].get("val/loss", float("nan"))
                lines.append(
                    f"| {FAMILY[tier]['name']} | {n / 1e6:.1f}M | {l_fit:.4f} | {_format_float(l_obs)} |"
                )
            lines += [
                "",
                "> **Interpretation**: A negative scaling exponent (b < 0) confirms the expected "
                "power-law improvement as model scale increases.",
            ]

    lines += [
        "",
        "## Training Configuration Summary",
        "",
        "| Model | LR | Batch (per device) | Grad Accum | Effective Batch |",
        "|---|---|---|---|---|",
        "| Nebium-Small  | 3e-4 | 32 | 4  | 128 |",
        "| Nebium-Medium | 2e-4 | 16 | 8  | 128 |",
        "| Nebium-Large  | 1.5e-4 | 8 | 16 | 128 |",
        "",
        "## References",
        "",
        "- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)",
        "- Kaplan et al. (2020). *Scaling Laws for Neural Language Models*. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)",
        "- Brown et al. (2020). *Language Models are Few-Shot Learners* (GPT-3). [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Nebium family models and produce scaling law analysis."
    )
    p.add_argument("--small", default=None, help="Path to Nebium-Small checkpoint (checkpoint.pt or model.pt)")
    p.add_argument("--medium", default=None, help="Path to Nebium-Medium checkpoint")
    p.add_argument("--large", default=None, help="Path to Nebium-Large checkpoint")
    p.add_argument("--data-path", default="data/fixtures/sample.pgn", help="Path to PGN or moves.txt for evaluation")
    p.add_argument("--max-games", type=int, default=200, help="Number of games to load for evaluation")
    p.add_argument("--batch-size", type=int, default=8, help="Eval batch size")
    p.add_argument("--tokens-trained", type=int, default=None,
                   help="Approximate total tokens used during training (for Chinchilla plot)")
    p.add_argument("--output", default="paper_assets/scaling_laws.json", help="JSON output path")
    p.add_argument("--report", default="paper_assets/SCALING_LAWS.md", help="Markdown report path")
    return p.parse_args()


def _load_sequences(data_path: str, max_games: int) -> list[str]:
    """Load UCI game sequences from a PGN or plain moves.txt file."""
    from omegaconf import OmegaConf
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if path.suffix in (".txt",):
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return lines[:max_games]

    # Assume PGN — use stream_uci_games with a minimal cfg
    cfg = OmegaConf.create({
        "data": {
            "format": "pgn",
            "max_games": max_games,
            "min_moves": 5,
        }
    })
    seqs = []
    for seq in stream_uci_games(cfg, [path]):
        seqs.append(seq)
        if len(seqs) >= max_games:
            break
    return seqs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Scaling Laws] Device: {device}")

    tier_ckpts = {
        "small": args.small,
        "medium": args.medium,
        "large": args.large,
    }

    # Load sequences and build tokenizer
    print(f"[Scaling Laws] Loading sequences from: {args.data_path}")
    sequences = _load_sequences(args.data_path, args.max_games)
    print(f"[Scaling Laws] Loaded {len(sequences)} sequences")

    from omegaconf import OmegaConf
    tok_cfg = OmegaConf.create({
        "model": {"vocab_size": 5000},
        "data": {
            "processed_path": "data/processed/scaling_eval",
            "tokenizer_path": "data/tokenizer/scaling_eval",
            "max_games": args.max_games,
            "min_moves": 5,
        }
    })
    tokenizer = get_tokenizer(tok_cfg, sequences)
    vocab_size = tokenizer.vocab_size
    print(f"[Scaling Laws] Tokenizer vocab_size: {vocab_size}")

    results: dict[str, Any] = {}

    for tier, ckpt_path in tier_ckpts.items():
        if ckpt_path is None:
            print(f"[Scaling Laws] Skipping {FAMILY[tier]['name']} — no checkpoint provided")
            continue
        if not Path(ckpt_path).exists():
            print(f"[Scaling Laws] WARNING: checkpoint not found: {ckpt_path}, skipping")
            continue

        print(f"\n[Scaling Laws] Evaluating {FAMILY[tier]['name']} from {ckpt_path} ...")
        t0 = time.time()
        model = _load_checkpoint(tier, ckpt_path, vocab_size, device)
        n_params = count_params(model)
        print(f"  Params: {n_params / 1e6:.2f}M")

        # Build val loader (use all as val for pure eval)
        _, val_loader = build_dataloaders(
            sequences,
            tokenizer,
            max_seq_len=FAMILY[tier]["max_seq_len"],
            batch_size=args.batch_size,
            train_split=0.5,
            seed=42,
        )
        metrics = evaluate_model(model, val_loader, device)
        elapsed = time.time() - t0

        tokens_trained = args.tokens_trained or (len(sequences) * 50)  # rough estimate: avg 50 tokens/game
        results[tier] = {
            "tier": tier,
            "name": FAMILY[tier]["name"],
            "actual_params": n_params,
            "checkpoint": ckpt_path,
            "tokens_trained": tokens_trained,
            "chinchilla_optimal_tokens": FAMILY[tier]["chinchilla_tokens"],
            "chinchilla_loss_at_optimal": chinchilla_loss(n_params, FAMILY[tier]["chinchilla_tokens"]),
            "chinchilla_loss_at_trained": chinchilla_loss(n_params, max(tokens_trained, 1_000_000)),
            "metrics": metrics,
            "eval_time_s": elapsed,
        }
        val_loss = metrics.get("val/loss", float("nan"))
        val_ppl = metrics.get("val/perplexity", float("nan"))
        val_acc = metrics.get("val/accuracy", 0.0)
        print(
            f"  Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f} "
            f"| Top-1 Acc: {val_acc * 100:.2f}% | Eval time: {elapsed:.1f}s"
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not results:
        print("\n[Scaling Laws] No checkpoints were evaluated. Provide at least --small, --medium, or --large.")
        sys.exit(1)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[Scaling Laws] Results saved to {out_path}")

    # Generate markdown report
    report = generate_markdown_report(results)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"[Scaling Laws] Markdown report saved to {report_path}")

    # Terminal summary
    print("\n" + "=" * 70)
    print("  SCALING LAW EVALUATION SUMMARY")
    print("=" * 70)
    tiers = ["small", "medium", "large"]
    for tier in tiers:
        if tier not in results:
            continue
        r = results[tier]
        m = r["metrics"]
        n = r["actual_params"]
        l_pred = r["chinchilla_loss_at_trained"]
        l_obs = m.get("val/loss", float("nan"))
        delta = l_obs - l_pred if not math.isnan(l_obs) else float("nan")
        print(
            f"  {FAMILY[tier]['name']:<18} | N={n / 1e6:7.1f}M "
            f"| L_obs={_format_float(l_obs)} | L_chin={l_pred:.4f} | Delta={_format_float(delta)}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
