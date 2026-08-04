from typing import Any

import pyarrow as pa
from pydantic import BaseModel

BUILD_PROFILE: str
""""debug" or "release" — benchmarks refuse a debug build."""

class DuckDBInferFn:
    """SQL specialized against frozen static tables, served bit-exact with DuckDB.

    Construction either succeeds with a function that matches DuckDB
    bit-for-bit — *with the declared `udfs` registered*, when there are any —
    or raises `ValueError` naming the unsupported construct. There is no
    third mode.
    """

    output_model: type[BaseModel] | None

    def __init__(
        self,
        sql: str,
        row_tables: dict[str, type[BaseModel]],
        static_tables: dict[str, pa.Table],
        udfs: list[Any] | None = None,
        output_model: type[BaseModel] | None = None,
        output: str | None = None,
        shape: str | None = None,
    ) -> None:
        """`udfs`: declared opaque scalar functions the SQL may call
        (DRAFT-22). Each object carries `name: str`, `takes: tuple[str, ...]`
        and `returns: tuple[str, ...]` in the engine type vocabulary
        ("i1" | "i64" | "f64" | "str"), and a scalar `__call__(*args)`
        receiving one plain Python value per argument (None for NULL) and
        returning a tuple matching `returns`, or None for an all-NULL
        result. An object with an `instances` attribute is a fitted
        transformer: its implicit leading argument is a nullable BIGINT
        instance id (never written in `takes`). Width-1 calls are scalar
        expressions; width-k calls are bare SELECT items emitting one
        `list | None` field — or, when the object declares
        `return_names: tuple[str, ...]` (one per return, TASK-63), field
        access over the call binds each addressed name to one lane of a
        single evaluation (`(f(...)).a` or `struct_extract(f(...), 'a')`,
        usable mid-expression; textually identical calls share the one
        evaluation, mirroring DuckDB's CSE). The oracle statement is
        parameterized, not
        weakened: the engine matches DuckDB running the same SQL with these
        same objects registered via `create_function`. UDFs must be
        deterministic.

        `output`: "model" (typed, default) or "dict".

        `shape`: the output-multiplicity contract, proven at build time --
        "map" (exactly one row out per row in; rejects WHERE and inner joins),
        "filter" (0 or 1, the default), or "many" (0..N; the only shape under
        which join multiplicity will build).
        """

    @property
    def shape(self) -> str:
        """The declared row-shape contract: "map", "filter", or "many"."""

    @property
    def output(self) -> str:
        """The output mode: "model" (typed, default) or "dict"."""

    @property
    def backend(self) -> str:
        """Which engine executes: "cranelift", "interpreter", or "constant"."""

    @property
    def boundary(self) -> str:
        """How rows cross the Python boundary: "marshaller" (generated at
        prepare), "generic" (env-pinned baseline), or "constant"."""

    def infer(
        self,
        tables: dict[str, list[Any]] | None = None,
        **kwargs: list[Any],
    ) -> list[BaseModel]: ...
    def infer_rows(self, rows: list[Any]) -> list[Any]:
        """Row objects in, row objects out -- the low-latency serving path."""

    def infer_arrow(self, batch: pa.Table) -> pa.Table:
        """`pa.Table` in, `pa.Table` out, with no per-value Python objects.

        Faster than `infer_rows` from roughly 1k rows per call; below that the
        fixed pyarrow API cost dominates and the row path wins.
        """
