"""SQLTransform — sklearn-compatible SQL-based feature transforms."""

from __future__ import annotations

from typing import Any

import datafusion
import pyarrow as pa

from sql_transform._interpreter import InferFn
from sql_transform._state import extract_state

__all__ = ["InferFn", "SQLTransform"]


class SQLTransform:
    """A transformer that applies SQL window-aggregate transforms.

    fit() runs the SQL on training data via DataFusion, extracts
    learned state (aggregate constants and partition lookups), and
    generates a Python function for single-row inference.

    transform() applies the transforms to batch data via DataFusion.

    Usage:
        t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM data")
        t.fit(train_table)
        out = t.transform(test_table)        # batch
        out_row = t._infer({"age": 42})      # single row
    """

    def __init__(self, sql: str) -> None:
        self._sql = sql
        self._state: dict[str, Any] = {}
        self._infer_fn: callable | None = None

    @classmethod
    def from_file(cls, path: str) -> SQLTransform:
        with open(path) as f:
            return cls(f.read())

    def fit(self, table: pa.Table, /) -> SQLTransform:
        ctx = datafusion.SessionContext()
        ctx.from_arrow(table, name="data")
        df = ctx.sql(self._sql)
        plan = df.logical_plan()

        self._state = extract_state(plan, ctx, "data")
        self._infer_fn = generate_infer_fn(plan, self._state)
        return self

    def transform(self, table: pa.Table, /) -> pa.Table:
        """Apply transforms to batch data using learned state.

        Iterates rows and calls the generated Python inference function.
        For large batches, this is slower than DataFusion but guarantees
        training-state transforms (not recomputed aggregates).
        """
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before transform()")
        rows = table.to_pylist()
        cols = table.column_names
        out_rows = [self._infer_fn({c: row[c] for c in cols}) for row in rows]
        return (
            pa.table({k: [r[k] for r in out_rows] for k in out_rows[0]})
            if out_rows
            else pa.table({})
        )

    def _infer(self, row: dict) -> dict:
        """Single-row inference using generated Python function."""
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before inference")
        return self._infer_fn(row)
