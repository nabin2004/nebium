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

import json
from src.evaluation.chess_metrics import generate_sample_games, evaluate_puzzles
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
):
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
    name = cfg.training.optimizer
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
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
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
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
            puzzle_metrics = evaluate_puzzles(model, tokenizer, device, puzzles_dataset)
            last_metrics.update(puzzle_metrics)
            puzzle_acc = puzzle_metrics.get("val/puzzle_accuracy", 0.0)

        logger.log_metrics(last_metrics, step=global_step)

        # Log samples table to WandB if supported
        if samples and hasattr(logger, "log_table"):
            table_cols = ["epoch", "prompt", "generated_moves", "legal_moves", "total_moves", "legal_rate"]
            table_rows = [
                [epoch + 1, s["prompt"], s["generated"], s["legal_moves"], s["total_moves"], s["legal_rate"]]
                for s in samples
            ]
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
            "val_loss": val_loss
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

    return last_metrics
