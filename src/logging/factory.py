from omegaconf import DictConfig

from src.logging.base import Logger
from src.logging.null_logger import NullLogger
from src.logging.wandb_logger import WandbLogger


def build_logger(cfg: DictConfig) -> Logger:
    backend = cfg.logging.backend
    if backend is None or backend == "null":
        return NullLogger()
    if backend == "wandb":
        return WandbLogger(cfg)
    raise ValueError(f"Unknown logging backend: {backend}")
