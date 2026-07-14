from typing import Any

class InferFn:
    def __init__(
        self,
        sql: str,
        row_tables: list[str],
        static_tables: dict[str, Any],
    ) -> None: ...
    def infer(
        self, tables: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]: ...
