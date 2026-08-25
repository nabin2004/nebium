import math

import torch
import torch.nn.functional as F


def next_token_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    valid = flat_labels != -100
    n_tokens = int(valid.sum().item())
    if n_tokens == 0:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "top5_accuracy": float("nan"),
            "perplexity": float("nan"),
            "n_tokens": 0,
        }

    valid_logits = flat_logits[valid]
    valid_labels = flat_labels[valid]
    loss_sum = float(F.cross_entropy(valid_logits, valid_labels, reduction="sum").item())
    preds = valid_logits.argmax(dim=-1)
    accuracy = float((preds == valid_labels).float().mean().item())
    k = min(5, vocab_size)
    topk = valid_logits.topk(k, dim=-1).indices
    top5 = float(topk.eq(valid_labels.unsqueeze(-1)).any(dim=-1).float().mean().item())
    mean_loss = loss_sum / n_tokens
    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "top5_accuracy": top5,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "n_tokens": n_tokens,
    }


def merge_metric_batches(batches: list[dict[str, float]]) -> dict[str, float]:
    n_tokens = sum(int(batch["n_tokens"]) for batch in batches)
    if n_tokens == 0:
        return {
            "val/loss": float("nan"),
            "val/accuracy": float("nan"),
            "val/top5_accuracy": float("nan"),
            "val/perplexity": float("nan"),
        }
    loss = sum(batch["loss"] * batch["n_tokens"] for batch in batches) / n_tokens
    accuracy = sum(batch["accuracy"] * batch["n_tokens"] for batch in batches) / n_tokens
    top5 = sum(batch["top5_accuracy"] * batch["n_tokens"] for batch in batches) / n_tokens
    return {
        "val/loss": loss,
        "val/accuracy": accuracy,
        "val/top5_accuracy": top5,
        "val/perplexity": math.exp(min(loss, 20.0)),
    }
