from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


class WandbLogger:
    def __init__(self, cfg: DictConfig) -> None:
        logging_cfg = cfg.logging
        self._run = wandb.init(
            project=logging_cfg.project,
            entity=logging_cfg.get("entity"),
            name=logging_cfg.get("run_name"),
            tags=list(logging_cfg.tags) if logging_cfg.get("tags") else None,
            notes=logging_cfg.get("notes"),
            mode=logging_cfg.get("mode", "online"),
        )

    def log_config(self, cfg: Any) -> None:
        config = OmegaConf.to_container(cfg, resolve=True)
        wandb.config.update(config, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        wandb.log(metrics, step=step)

    def log_table(
        self,
        table_name: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        table = wandb.Table(columns=columns, data=data)
        wandb.log({table_name: table}, step=step)

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        wandb.log({key: wandb.Html(f"<pre style='font-family: monospace;'>{text}</pre>")}, step=step)

    def watch_model(self, model: Any) -> None:
        wandb.watch(model)

    def finish(self) -> None:
        wandb.finish()
