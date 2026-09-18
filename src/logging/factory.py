"""
Logger instantiation factory for Nebium.
"""

from omegaconf import DictConfig

from src.logging.base import Logger
from src.logging.null_logger import NullLogger
from src.logging.wandb_logger import WandbLogger


def build_logger(cfg: DictConfig) -> Logger:
    """
    Instantiates the configured experiment logging backend.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Logger protocol implementation (WandbLogger or NullLogger).

    Raises:
        ValueError: If `backend` is not recognized.
    """
    backend = cfg.logging.backend
    if backend is None or backend == "null":
        return NullLogger()
    if backend == "wandb":
        return WandbLogger(cfg)
    raise ValueError(f"Unknown logging backend: {backend}")

