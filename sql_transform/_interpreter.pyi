from typing import Any

import pyarrow as pa
from pydantic import BaseModel

class InferFn:
    output_model: type[BaseModel]

    def __init__(
        self,
        sql: str,
        row_tables: dict[str, type[BaseModel]],
        static_tables: dict[str, pa.Table],
        output_model: type[BaseModel] | None = None,
    ) -> None: ...
    def infer(
        self,
        tables: dict[str, list[Any]] | None = None,
        **kwargs: list[Any],
    ) -> list[BaseModel]: ...
