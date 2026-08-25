import hydra
from omegaconf import DictConfig

from src.logging.factory import build_logger


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger = build_logger(cfg)
    try:
        logger.log_config(cfg)
        # Training loop: logger.watch_model(model); logger.log_metrics(...)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
