from typing import Any, Protocol


class Logger(Protocol):
    def log_config(self, cfg: Any) -> None: ...

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

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
