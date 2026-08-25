import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.evaluation.metrics import merge_metric_batches, next_token_metrics
from src.logging.base import Logger


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
    global_step = 0
    last_metrics: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(int(cfg.training.epochs)):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.training.epochs}")
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast("cuda", enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                loss = loss / accum
            scaler.scale(loss).backward()
            running += float(loss.item()) * accum

            if step % accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                lr = optimizer.param_groups[0]["lr"]
                logger.log_metrics({"train/loss": running / max(1, step), "lr": lr}, step=global_step)
                progress.set_postfix(loss=running / max(1, step), lr=lr)

        metrics = evaluate(model, val_loader, device, use_amp)
        last_metrics = {**metrics, "epoch": epoch + 1}
        logger.log_metrics(last_metrics, step=global_step)
        progress.set_postfix(
            loss=running / max(1, len(train_loader)),
            acc=metrics["val/accuracy"],
            top5=metrics["val/top5_accuracy"],
            ppl=metrics["val/perplexity"],
        )
    return last_metrics
