"""SQLTransform — sklearn-compatible SQL-based feature transforms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import datafusion
import pyarrow as pa
from pydantic import BaseModel

from sql_transform._batch import run_batch
from sql_transform._interpreter import InferFn
from sql_transform._rewrite import rewrite_sql
from sql_transform._schema import synthesize_this_model
from sql_transform._sql import find_window_aggregates, parse_and_validate
from sql_transform._state import build_state_tables

__all__ = ["InferFn", "SQLTransform"]


def _to_namespace(row: dict[str, Any] | BaseModel) -> SimpleNamespace:
    """Normalize an inference input row (dict or Pydantic model) into the
    SimpleNamespace of attributes the Rust InferFn reads."""
    if isinstance(row, BaseModel):
        return SimpleNamespace(**row.model_dump())
    return SimpleNamespace(**row)


class SQLTransform:
    """A transformer that applies SQL window-aggregate transforms.

    fit() runs the SQL on training data via DataFusion to extract window
    aggregate state, rewrites the SQL into plain-column-reference form,
    and builds a Rust InferFn for evaluation.

    transform() applies the transforms to batch data via DataFusion.
    infer()/infer_batch() apply them row-at-a-time via the Rust InferFn.

    Usage:
        t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
        t.fit(train_table)
        out = t.transform(test_table)        # batch
        out_row = t.infer({"age": 42})       # single row
    """

    def __init__(self, sql: str) -> None:
        self._sql = sql
        self._state_tables: dict[str, pa.Table] | None = None
        self._rewritten_sql: str | None = None
        self._infer_fn: InferFn | None = None

    @classmethod
    def from_file(cls, path: str) -> SQLTransform:
        with open(path) as f:
            return cls(f.read())

    def fit(
        self,
        table: pa.Table,
        /,
        this_model: type[BaseModel] | None = None,
    ) -> SQLTransform:
        this_model = this_model or synthesize_this_model(table.schema)

        tree = parse_and_validate(self._sql)
        windows = find_window_aggregates(tree)

        ctx = datafusion.SessionContext()
        ctx.from_arrow(table, name="__THIS__")

        self._state_tables = build_state_tables(windows, ctx, "__THIS__")
        self._rewritten_sql = rewrite_sql(tree, windows)
        self._infer_fn = InferFn(
            self._rewritten_sql,
            row_tables={"__THIS__": this_model},
            static_tables=self._state_tables,
        )
        return self

    def transform(self, table: pa.Table, /) -> pa.Table:
        """Batch-transform `table` through DataFusion using the frozen fit-time
        state. Runs the rewritten SQL (`__THIS__` LEFT JOINed to the per-partition
        state tables) vectorized; returns a pyarrow Table with rows in input order.
        Use infer()/infer_batch() for low-latency row-at-a-time inference through
        the Rust engine instead."""
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before transform")
        return run_batch(self._rewritten_sql, table, self._state_tables)

    def infer(self, row: dict[str, Any] | BaseModel, /) -> BaseModel:
        """Single-row inference through the Rust InferFn against the frozen
        state. Accepts a dict or a Pydantic model; returns the typed output
        model instance."""
        return self.infer_batch([row])[0]

    def infer_batch(self, rows: list[dict[str, Any] | BaseModel], /) -> list[BaseModel]:
        """Many-rows inference through the Rust InferFn against the frozen
        state. Accepts dicts and/or Pydantic models; returns a list of typed
        output model instances."""
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before inference")
        this_rows = [_to_namespace(row) for row in rows]
        return self._infer_fn.infer({"__THIS__": this_rows})
