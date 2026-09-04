import math

import torch
import torch.nn.functional as F


def next_token_metrics(
    logits: torch.Tensor, 
    labels: torch.Tensor,
    special_token_ids: list[int] | None = None,
) -> dict[str, float]:
    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    valid = flat_labels != -100
    n_tokens = int(valid.sum().item())
    
    empty_metrics = {
        "loss": float("nan"),
        "accuracy": float("nan"),
        "top5_accuracy": float("nan"),
        "perplexity": float("nan"),
        "n_tokens": 0,
    }
    
    if n_tokens == 0:
        return empty_metrics

    valid_logits = flat_logits[valid]
    valid_labels = flat_labels[valid]
    loss_sum = float(F.cross_entropy(valid_logits, valid_labels, reduction="sum").item())
    preds = valid_logits.argmax(dim=-1)
    accuracy = float((preds == valid_labels).float().mean().item())
    k = min(5, vocab_size)
    topk = valid_logits.topk(k, dim=-1).indices
    top5 = float(topk.eq(valid_labels.unsqueeze(-1)).any(dim=-1).float().mean().item())
    mean_loss = loss_sum / n_tokens
    
    metrics = {
        "loss": mean_loss,
        "accuracy": accuracy,
        "top5_accuracy": top5,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "n_tokens": n_tokens,
    }
    
    if special_token_ids is not None:
        is_special = torch.isin(valid_labels, torch.tensor(special_token_ids, device=valid_labels.device))
        
        n_special = int(is_special.sum().item())
        n_move = n_tokens - n_special
        
        if n_special > 0:
            special_loss = float(F.cross_entropy(valid_logits[is_special], valid_labels[is_special], reduction="sum").item()) / n_special
            metrics["special_loss"] = special_loss
            metrics["special_perplexity"] = math.exp(min(special_loss, 20.0))
            metrics["n_special"] = n_special
        
        if n_move > 0:
            move_loss = float(F.cross_entropy(valid_logits[~is_special], valid_labels[~is_special], reduction="sum").item()) / n_move
            metrics["move_loss"] = move_loss
            metrics["move_perplexity"] = math.exp(min(move_loss, 20.0))
            metrics["n_move"] = n_move
            
    return metrics


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
    
    result = {
        "val/loss": loss,
        "val/accuracy": accuracy,
        "val/top5_accuracy": top5,
        "val/perplexity": math.exp(min(loss, 20.0)),
    }
    
    n_special = sum(int(batch.get("n_special", 0)) for batch in batches)
    if n_special > 0:
        special_loss = sum(batch["special_loss"] * batch["n_special"] for batch in batches if "n_special" in batch) / n_special
        result["val/special_loss"] = special_loss
        result["val/special_perplexity"] = math.exp(min(special_loss, 20.0))
        
    n_move = sum(int(batch.get("n_move", 0)) for batch in batches)
    if n_move > 0:
        move_loss = sum(batch["move_loss"] * batch["n_move"] for batch in batches if "n_move" in batch) / n_move
        result["val/move_loss"] = move_loss
        result["val/move_perplexity"] = math.exp(min(move_loss, 20.0))
        
    return result
