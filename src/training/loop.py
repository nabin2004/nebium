"""
Model training, evaluation, and checkpoint orchestration loop for Nebium.

Implements AMP mixed precision training with AdamW, linear warmup cosine decay,
gradient norm clipping, validation loss aggregation, and puzzle evaluation benchmarks.
"""

import json
import math
import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.evaluation.chess_metrics import evaluate_puzzles, generate_sample_games
from src.evaluation.metrics import merge_metric_batches, next_token_metrics
from src.logging.base import Logger


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    epoch: int,
    global_step: int,
    scaler: GradScaler,
) -> None:
    """
    Serializes complete model weights, optimizer momentum, learning rate scheduler,
    and AMP gradient scaler states to disk.

    Args:
        path: Filepath destination for the checkpoint (.pt).
        model: Training neural network model.
        optimizer: Optimizer instance.
        scheduler: Optional learning rate scheduler.
        epoch: Current epoch index.
        global_step: Global optimization step counter.
        scaler: AMP gradient scaler.
    """
    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "scaler": scaler.state_dict() if scaler else None,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    scaler: GradScaler,
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    if hasattr(model, "module"):
        model.module.load_state_dict(state["model"])
    else:
        model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if "scaler" in state and state["scaler"] and scaler:
        scaler.load_state_dict(state["scaler"])
    return state.get("epoch", 0), state.get("global_step", 0)


def build_optimizer(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    name = str(cfg.training.optimizer).lower()
    lr = float(cfg.training.learning_rate)
    weight_decay = float(cfg.training.get("weight_decay", 0.1))

    if name in ("adamw_8bit", "bnb_8bit", "adamw8bit"):
        try:
            import bitsandbytes as bnb
            print("[Optimizer] Using bitsandbytes 8-bit AdamW (AdamW8bit)")
            return bnb.optim.AdamW8bit(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        except ImportError:
            print("[Optimizer] Note: bitsandbytes not installed; falling back to memory-optimized AdamW")
            name = "adamw"

    if name == "adamw":
        # On CUDA, prioritize fused=True or foreach=False to prevent large temporary memory
        # allocations from PyTorch's default _multi_tensor_adam (_foreach_sqrt)
        is_cuda = False
        try:
            is_cuda = next(model.parameters()).is_cuda
        except (StopIteration, Exception):
            pass

        if is_cuda:
            try:
                # fused=True executes an in-place C++/CUDA kernel without allocating intermediate tensor copies
                return torch.optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    fused=True,
                )
            except Exception as exc:
                print(f"[Optimizer] Note: fused=True not supported ({exc}); falling back to foreach=False")
                return torch.optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    foreach=False,
                )
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig, total_steps: int) -> LambdaLR | None:
    if cfg.training.lr_scheduler != "cosine":
        return None
    warmup_steps = int(cfg.training.warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_loader, device: torch.device, use_amp: bool) -> dict[str, float]:
    if len(val_loader) == 0:
        return {
            "val/loss": float("nan"),
            "val/accuracy": float("nan"),
            "val/top5_accuracy": float("nan"),
            "val/perplexity": float("nan"),
        }
    model.eval()
    batches: list[dict[str, float]] = []
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast("cuda", enabled=use_amp):
            logits = model(input_ids, attention_mask)
        batches.append(next_token_metrics(logits.float(), labels))
    return merge_metric_batches(batches)


def run_training(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    cfg: DictConfig,
    logger: Logger,
    tokenizer: Any = None,
) -> dict[str, float]:
    device = _device()
    model = model.to(device)

    # Enable activation (gradient) checkpointing to reduce VRAM
    if bool(cfg.training.get("gradient_checkpointing", False)):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("[Training] Enabled gradient (activation) checkpointing.")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_amp = cfg.training.mixed_precision == "fp16" and device.type == "cuda"
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / int(cfg.training.gradient_accumulation_steps)))
    total_steps = int(cfg.training.epochs) * steps_per_epoch
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = GradScaler("cuda", enabled=use_amp)
    accum = int(cfg.training.gradient_accumulation_steps)

    logger.watch_model(model)
    start_epoch = 0
    global_step = 0

    if cfg.training.get("resume_from"):
        resume_path = cfg.training.resume_from
        if os.path.exists(resume_path):
            start_epoch, global_step = load_checkpoint(resume_path, model, optimizer, scheduler, scaler)
            print(f"Resumed from {resume_path} at epoch {start_epoch}, step {global_step}")
        else:
            print(f"Checkpoint not found at {resume_path}, starting from scratch.")

    puzzles_dataset = []
    if cfg.training.get("eval_puzzles", False):
        puzzle_path = cfg.training.get("puzzle_path", "data/fixtures/puzzles.jsonl")
        if os.path.exists(puzzle_path):
            with open(puzzle_path, "r", encoding="utf-8") as f:
                puzzles_dataset = [json.loads(line) for line in f if line.strip()]
            if bool(cfg.get("smoke_test", False)):
                puzzles_dataset = puzzles_dataset[:5]
            print(f"Loaded {len(puzzles_dataset)} puzzles for evaluation.")
        else:
            print(f"Puzzle dataset not found at {puzzle_path}")

    last_metrics: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)

    best_val_loss = float("inf")
    patience_counter = 0
    training_history_list = []

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        model.train()
        running = 0.0
        start_time = time.time()
        tokens_processed = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.training.epochs}")
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Count valid tokens for throughput
            tokens_processed += int((labels != -100).sum().item())

            with autocast("cuda", enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                loss = loss / accum
            scaler.scale(loss).backward()
            running += float(loss.item()) * accum

            if step % accum == 0 or step == len(train_loader):
                grad_clip = float(cfg.training.get("gradient_clipping", 1.0))
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if torch.cuda.is_available() and accum >= 32 and global_step % 5 == 0:
                    torch.cuda.empty_cache()
                lr = optimizer.param_groups[0]["lr"]
                logger.log_metrics({"train/loss": running / max(1, step), "lr": lr}, step=global_step)
                progress.set_postfix(loss=running / max(1, step), lr=lr)

        elapsed = time.time() - start_time
        metrics = evaluate(model, val_loader, device, use_amp)

        last_metrics = {
            **metrics,
            "epoch": epoch + 1,
            "train/throughput_tokens_sec": tokens_processed / elapsed if elapsed > 0 else 0.0,
        }

        if torch.cuda.is_available():
            last_metrics["train/gpu_mem_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
            torch.cuda.reset_peak_memory_stats()

        # Generate sample chess moves if tokenizer is provided
        samples: list[dict[str, Any]] = []
        overall_legal_rate = 0.0
        if tokenizer is not None and bool(cfg.training.get("eval_samples", True)):
            raw_prompts = cfg.training.get("sample_prompts", ["", "e2e4", "d2d4"])
            sample_prompts = list(raw_prompts) if raw_prompts else ["", "e2e4", "d2d4"]
            max_moves = int(cfg.training.get("sample_max_moves", 20))
            temp = float(cfg.training.get("sample_temperature", 0.7))
            samples = generate_sample_games(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=sample_prompts,
                max_moves=max_moves,
                temperature=temp,
            )
            total_legal = sum(s["legal_moves"] for s in samples)
            total_moves = sum(s["total_moves"] for s in samples)
            overall_legal_rate = (total_legal / total_moves) if total_moves > 0 else 0.0
            last_metrics["val/legal_move_rate"] = overall_legal_rate

        # Evaluate puzzles if loaded
        puzzle_acc = 0.0
        if puzzles_dataset and tokenizer is not None:
            puzzle_metrics, puzzle_details = evaluate_puzzles(
                model, tokenizer, device, puzzles_dataset, return_details=True
            )
            last_metrics.update(puzzle_metrics)
            puzzle_acc = puzzle_metrics.get("val/puzzle_accuracy", 0.0)

            if puzzle_details and hasattr(logger, "log_table"):
                p_cols = ["epoch", "puzzle_id", "rating", "bracket", "prompt", "solution", "predicted", "status"]
                p_rows = [
                    [
                        epoch + 1,
                        p["puzzle_id"],
                        p["rating"],
                        p["bracket"],
                        p["prompt"],
                        p["solution"],
                        p["predicted"],
                        "✅ Correct" if p["is_correct"] else "❌ Failed",
                    ]
                    for p in puzzle_details
                ]
                logger.log_table("eval/puzzle_benchmarks", p_cols, p_rows, step=global_step)

        logger.log_metrics(last_metrics, step=global_step)

        # Log samples table to WandB if supported
        if samples and hasattr(logger, "log_table"):
            import chess
            table_cols = [
                "epoch",
                "prompt",
                "generated_moves",
                "legality_status",
                "legal_moves",
                "total_moves",
                "legal_rate_pct",
                "stop_reason",
                "final_fen",
            ]
            table_rows = []
            for s in samples:
                tot = s["total_moves"]
                leg = s["legal_moves"]
                rate = s["legal_rate"]
                if rate == 1.0 and tot > 0:
                    status = "✅ 100% Legal"
                elif rate >= 0.5:
                    status = "⚠️ Partial Legality"
                elif tot == 0:
                    status = "ℹ️ No Moves"
                else:
                    status = "❌ Illegal / Diverged"

                board = chess.Board()
                full_moves = s.get("full_game", "").split()
                for m_str in full_moves:
                    try:
                        mv = chess.Move.from_uci(m_str)
                        if mv in board.legal_moves:
                            board.push(mv)
                        else:
                            break
                    except Exception:
                        break

                table_rows.append([
                    epoch + 1,
                    s["prompt"],
                    s["generated"],
                    status,
                    leg,
                    tot,
                    round(rate * 100, 1),
                    s.get("stop_reason", "unknown"),
                    board.fen(),
                ])
            logger.log_table("eval/sample_generations", table_cols, table_rows, step=global_step)

        # Terminal inspection block after each loop/epoch
        train_loss_epoch = running / max(1, len(train_loader))
        val_loss = metrics.get("val/loss", float("nan"))
        val_acc = metrics.get("val/accuracy", float("nan"))
        val_top5 = metrics.get("val/top5_accuracy", float("nan"))
        val_ppl = metrics.get("val/perplexity", float("nan"))
        lr = optimizer.param_groups[0]["lr"]
        throughput = tokens_processed / elapsed if elapsed > 0 else 0.0
        
        training_history_list.append({
            "epoch": epoch + 1,
            "train_loss": train_loss_epoch,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "val_accuracy": val_acc,
            "val_top5_accuracy": val_top5,
            "legal_move_rate": overall_legal_rate,
            "puzzle_acc": puzzle_acc,
        })

        print("\n" + "=" * 78)
        print(f"  [EPOCH {epoch + 1}/{cfg.training.epochs} EVALUATION SUMMARY]")
        print("-" * 78)
        print(f"  Train Loss: {train_loss_epoch:.4f}  |  Val Loss: {val_loss:.4f}  |  Val PPL: {val_ppl:.2f}")
        print(f"  Top-1 Acc:  {val_acc * 100:.2f}%  |  Top-5 Acc: {val_top5 * 100:.2f}%  |  Legal Moves: {overall_legal_rate * 100:.2f}%")
        print(f"  Throughput: {throughput:,.0f} tok/s  |  LR: {lr:.2e}")
        if puzzles_dataset:
            print(f"  Puzzle Acc: {puzzle_acc * 100:.2f}%")
        if samples:
            print("-" * 78)
            print("  SAMPLE GENERATIONS (Epoch Progress Inspection):")
            for idx, s in enumerate(samples, start=1):
                prompt_disp = s["prompt"] if s["prompt"] else "[empty board]"
                gen_disp = s["generated"] if s["generated"] else "[no moves generated]"
                pct_disp = f"{s['legal_rate'] * 100:.1f}%" if s["total_moves"] > 0 else "N/A"
                print(f"    ({idx}) Prompt: '{prompt_disp}'")
                print(f"        -> Moves ({s['legal_moves']}/{s['total_moves']} legal, {pct_disp}): {gen_disp}")
        print("=" * 78 + "\n")

        progress.set_postfix(
            loss=running / max(1, len(train_loader)),
            acc=metrics["val/accuracy"],
            top5=metrics["val/top5_accuracy"],
            ppl=metrics["val/perplexity"],
        )
        save_checkpoint("checkpoint.pt", model, optimizer, scheduler, epoch + 1, global_step, scaler)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint("best_model.pt", model, optimizer, scheduler, epoch + 1, global_step, scaler)
        else:
            patience_counter += 1
            patience = int(cfg.training.get("early_stopping_patience", 5))
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    with open("training_history.json", "w") as f:
        json.dump(training_history_list, f, indent=4)

    # -----------------------------------------------------------------------
    # Publication-Grade Visualizations & WandB Artifact Logging
    # -----------------------------------------------------------------------
    try:
        from src.logging.figures import (
            plot_training_dynamics,
            plot_accuracy_and_legality,
            plot_scaling_law_alignment,
            save_publication_figures,
        )

        n_params = sum(p.numel() for p in model.parameters())
        tier_label = "Nebium"
        tier_slug = "small"
        if hasattr(cfg, "model") and hasattr(cfg.model, "d_model"):
            d = int(cfg.model.d_model)
            n = int(cfg.model.n_layers)
            if d == 768 and n == 12:
                tier_label = "Nebium-Small (117M)"
                tier_slug = "small"
            elif d == 1024 and n == 24:
                tier_label = "Nebium-Medium (345M)"
                tier_slug = "medium"
            elif d == 1280 and n == 36:
                tier_label = "Nebium-Large (762M)"
                tier_slug = "large"

        # 1. Log high-resolution publication figures to WandB
        if training_history_list:
            fig_dyn = plot_training_dynamics(training_history_list, tier_name=tier_label)
            logger.log_figure("publication/training_dynamics", fig_dyn, step=global_step)

            fig_acc = plot_accuracy_and_legality(training_history_list, tier_name=tier_label)
            logger.log_figure("publication/accuracy_and_legality", fig_acc, step=global_step)

        best_loss_val = best_val_loss if not math.isinf(best_val_loss) else last_metrics.get("val/loss", float("nan"))
        fig_scale = plot_scaling_law_alignment(
            empirical_params=n_params,
            empirical_loss=best_loss_val,
            current_tier=tier_label,
        )
        logger.log_figure("publication/scaling_laws", fig_scale, step=global_step)

        # 2. Persist publication assets to disk
        save_publication_figures(
            history=training_history_list,
            empirical_params=n_params,
            empirical_loss=best_loss_val,
            tier_name=tier_label,
            output_dir="paper_assets",
        )

        # 3. Log versioned WandB artifacts
        if os.path.exists("best_model.pt"):
            logger.log_artifact(
                "best_model.pt",
                name=f"nebium-{tier_slug}-best-model",
                type="model",
                metadata=last_metrics,
            )
        if os.path.exists("nebium.gguf"):
            logger.log_artifact(
                "nebium.gguf",
                name=f"nebium-{tier_slug}-gguf",
                type="gguf",
                metadata={"precision": str(cfg.training.get("gguf_precision", "fp16"))},
            )
        if os.path.exists("paper_assets"):
            logger.log_artifact(
                "paper_assets",
                name=f"nebium-{tier_slug}-figures",
                type="publication_figures",
                metadata={"dpi": 300, "tier": tier_slug},
            )

        # 4. Record publication-grade summary metrics
        logger.log_summary({
            "best_val_loss": best_loss_val,
            "final_val_loss": last_metrics.get("val/loss", float("nan")),
            "final_val_perplexity": last_metrics.get("val/perplexity", float("nan")),
            "final_top1_accuracy": last_metrics.get("val/accuracy", float("nan")),
            "final_top5_accuracy": last_metrics.get("val/top5_accuracy", float("nan")),
            "final_legal_move_rate": last_metrics.get("val/legal_move_rate", float("nan")),
            "puzzle_accuracy": last_metrics.get("val/puzzle_accuracy", float("nan")),
            "total_parameters": n_params,
            "tier": tier_slug,
        })
    except Exception as exc:
        print(f"[Publication Figures] Note: Figure logging caught: {exc}")

    return last_metrics
