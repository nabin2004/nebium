"""
Weights & Biases experiment tracking logger for Nebium.

Implements publication-grade experiment tracking with structured metadata,
custom metric summaries and optimization targets, publication figures,
interactive evaluation tables, and artifact versioning.
"""

from pathlib import Path
import time
from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


class WandbLogger:
    """
    Weights & Biases logger integrating experiment metadata, metrics, tables,
    publication figures, and versioned artifacts.
    """

    def __init__(self, cfg: DictConfig) -> None:
        logging_cfg = cfg.logging

        # Infer model tier if available
        tier = "base"
        if hasattr(cfg, "model") and hasattr(cfg.model, "d_model"):
            d = int(cfg.model.d_model)
            n = int(cfg.model.n_layers)
            if d == 768 and n == 12:
                tier = "small"
            elif d == 1024 and n == 24:
                tier = "medium"
            elif d == 1280 and n == 36:
                tier = "large"

        run_name = logging_cfg.get("run_name")
        if not run_name:
            run_name = f"nebium-{tier}-{time.strftime('%Y%m%d-%H%M%S')}"

        # Consolidate tags
        tags = list(logging_cfg.tags) if logging_cfg.get("tags") else []
        default_tags = ["nebium", f"nebium-{tier}", "transformer", "publication-grade"]
        for t in default_tags:
            if t not in tags:
                tags.append(t)

        self._run = wandb.init(
            project=logging_cfg.project,
            entity=logging_cfg.get("entity"),
            name=run_name,
            tags=tags,
            notes=logging_cfg.get("notes"),
            mode=logging_cfg.get("mode", "online"),
        )
        self.define_metrics()

    def define_metrics(self) -> None:
        """
        Registers WandB metric definitions, setting standard step metric bindings
        and optimization targets for publication-grade visualization.
        """
        if not wandb.run:
            return

        # Define step indices
        wandb.define_metric("step")
        wandb.define_metric("epoch")

        # Step metric bindings
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("val/*", step_metric="step")
        wandb.define_metric("eval/*", step_metric="step")

        # Metric goals and summary aggregators
        wandb.define_metric("val/loss", summary="min,mean", goal="minimize")
        wandb.define_metric("val/perplexity", summary="min", goal="minimize")
        wandb.define_metric("val/accuracy", summary="max", goal="maximize")
        wandb.define_metric("val/top5_accuracy", summary="max", goal="maximize")
        wandb.define_metric("val/legal_move_rate", summary="max", goal="maximize")
        wandb.define_metric("val/puzzle_accuracy", summary="max", goal="maximize")
        wandb.define_metric("train/loss", summary="min,mean", goal="minimize")
        wandb.define_metric("train/throughput_tokens_sec", summary="mean")

    def log_config(self, cfg: Any) -> None:
        config = OmegaConf.to_container(cfg, resolve=True)
        if wandb.run:
            wandb.config.update(config, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if wandb.run:
            wandb.log(metrics, step=step)

    def log_table(
        self,
        table_name: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        if wandb.run:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({table_name: table}, step=step)

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        if wandb.run:
            wandb.log({key: wandb.Html(f"<pre style='font-family: monospace;'>{text}</pre>")}, step=step)

    def log_figure(self, key: str, figure: Any, step: int | None = None) -> None:
        """Logs a Matplotlib figure as an image to WandB."""
        if wandb.run:
            wandb.log({key: wandb.Image(figure)}, step=step)

    def log_artifact(
        self,
        artifact_path: str | Path,
        name: str,
        type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Logs a local file or directory as an official versioned WandB artifact."""
        if not wandb.run:
            return
        path = Path(artifact_path)
        if not path.exists():
            return
        artifact = wandb.Artifact(name=name, type=type, metadata=metadata or {})
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        wandb.log_artifact(artifact)

    def log_summary(self, summary_metrics: dict[str, Any]) -> None:
        """Populates or updates the WandB run summary with key publication figures."""
        if wandb.run:
            wandb.run.summary.update(summary_metrics)

    def watch_model(self, model: Any) -> None:
        if wandb.run:
            wandb.watch(model)

    def finish(self) -> None:
        if wandb.run:
            wandb.finish()
