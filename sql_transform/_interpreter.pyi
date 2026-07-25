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
        transformers: dict[str, tuple[object, pa.Schema]] | None = None,
    ) -> None: ...
    def infer(
        self,
        tables: dict[str, list[Any]] | None = None,
        **kwargs: list[Any],
    ) -> list[BaseModel]: ...

class DuckDBInferFn:
    """DuckDB-semantics interpreter: the same API as `InferFn` without the
    transformer callout.

    Stub -- the SQL is parsed in the DuckDB dialect and nothing else; no output
    model is derived and `infer()` raises `NotImplementedError`.
    """

    output_model: type[BaseModel] | None

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
