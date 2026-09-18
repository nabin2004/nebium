"""
Abstract logging protocol for Nebium.

Defines the logging interface adhered to by WandBLogger and NullLogger.
"""

from typing import Any, Protocol


class Logger(Protocol):
    """
    Structural subtyping protocol for training metrics and artifact loggers.
    """

    def log_config(self, cfg: Any) -> None:
        """Logs experiment hyperparameters and run configuration."""
        ...

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Logs scalar numerical metrics at a given training step."""
        ...


    def log_table(
        self,
        table_name: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None: ...

    def log_text(self, key: str, text: str, step: int | None = None) -> None: ...

    def watch_model(self, model: Any) -> None: ...

    def finish(self) -> None: ...
