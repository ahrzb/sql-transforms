from typing import Any

from pydantic import BaseModel

class InferFn:
    output_model: type[BaseModel]

    def __init__(
        self,
        sql: str,
        row_tables: dict[str, type[BaseModel]],
        static_tables: dict[str, Any],
        output_model: type[BaseModel] | None = None,
    ) -> None: ...
    def infer(self, tables: dict[str, list[BaseModel]]) -> list[BaseModel]: ...
