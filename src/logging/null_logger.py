"""
No-op logger implementation for headless and testing execution.
"""

from typing import Any


class NullLogger:
    """
    A silent logger that consumes calls without performing I/O or network requests.
    """

    def log_config(self, cfg: Any) -> None:

        return None

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    def log_table(
        self,
        table_name: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        return None

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        return None

    def log_figure(self, key: str, figure: Any, step: int | None = None) -> None:
        return None

    def log_artifact(
        self,
        artifact_path: str | Any,
        name: str,
        type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return None

    def log_summary(self, summary_metrics: dict[str, Any]) -> None:
        return None

    def define_metrics(self) -> None:
        return None

    def watch_model(self, model: Any) -> None:
        return None

    def finish(self) -> None:
        return None
