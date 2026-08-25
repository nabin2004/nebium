from typing import Any


class NullLogger:
    def log_config(self, cfg: Any) -> None:
        return None

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    def watch_model(self, model: Any) -> None:
        return None

    def finish(self) -> None:
        return None
