import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import os
import time

from src.evaluation.metrics import merge_metric_batches, next_token_metrics
from src.logging.base import Logger


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: LambdaLR | None, epoch: int, global_step: int, scaler: GradScaler):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "scaler": scaler.state_dict()
    }
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: LambdaLR | None, scaler: GradScaler) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state["scheduler"]:
        scheduler.load_state_dict(state["scheduler"])
    if "scaler" in state and scaler:
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


def run_training(model: torch.nn.Module, train_loader, val_loader, cfg: DictConfig, logger: Logger) -> dict[str, float]:
    device = _device()
    model = model.to(device)
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

    last_metrics: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)

    best_val_loss = float("inf")
    patience_counter = 0

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
                if cfg.training.gradient_clipping > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clipping)
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
            last_metrics["train/gpu_mem_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()
            
        logger.log_metrics(last_metrics, step=global_step)
        progress.set_postfix(
            loss=running / max(1, len(train_loader)),
            acc=metrics["val/accuracy"],
            top5=metrics["val/top5_accuracy"],
            ppl=metrics["val/perplexity"],
        )
        save_checkpoint("checkpoint.pt", model, optimizer, scheduler, epoch + 1, global_step, scaler)

        val_loss = metrics.get("val/loss", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint("best_model.pt", model, optimizer, scheduler, epoch + 1, global_step, scaler)
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    return last_metrics
