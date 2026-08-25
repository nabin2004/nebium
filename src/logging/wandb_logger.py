from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


class WandbLogger:
    def __init__(self, cfg: DictConfig) -> None:
        logging_cfg = cfg.logging
        self._run = wandb.init(
            project=logging_cfg.project,
            entity=logging_cfg.entity,
            name=logging_cfg.run_name,
            tags=list(logging_cfg.tags) if logging_cfg.tags else None,
            notes=logging_cfg.notes,
            mode=logging_cfg.mode,
        )

    def log_config(self, cfg: Any) -> None:
        config = OmegaConf.to_container(cfg, resolve=True)
        wandb.config.update(config, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        wandb.log(metrics, step=step)

    def watch_model(self, model: Any) -> None:
        wandb.watch(model)

    def finish(self) -> None:
        wandb.finish()
