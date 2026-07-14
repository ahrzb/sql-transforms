"""SQLTransform — sklearn-compatible SQL-based feature transforms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import datafusion
import pyarrow as pa
from pydantic import BaseModel

from sql_transform._codegen import rewrite_sql
from sql_transform._interpreter import InferFn
from sql_transform._schema import synthesize_this_model
from sql_transform._state import extract_state

__all__ = ["InferFn", "SQLTransform"]


class SQLTransform:
    """A transformer that applies SQL window-aggregate transforms.

    fit() runs the SQL on training data via DataFusion to extract window
    aggregate state, rewrites the SQL into plain-column-reference form,
    and builds a Rust InferFn for evaluation.

    transform() applies the transforms to batch data via InferFn.

    Usage:
        t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
        t.fit(train_table)
        out = t.transform(test_table)        # batch
        out_row = t._infer({"age": 42})      # single row
    """

    def __init__(self, sql: str) -> None:
        self._sql = sql
        self._state: BaseModel | None = None
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

        ctx = datafusion.SessionContext()
        ctx.from_arrow(table, name="__THIS__")
        df = ctx.sql(self._sql)
        plan = df.logical_plan()

        self._state = extract_state(plan, ctx, "__THIS__")
        rewritten_sql = rewrite_sql(plan)
        self._infer_fn = InferFn(
            rewritten_sql,
            row_tables={"__THIS__": this_model, "__STATE__": type(self._state)},
            static_tables={},
        )
        return self

    def transform(self, table: pa.Table, /) -> pa.Table:
        """Apply transforms to batch data using learned state, via InferFn."""
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before transform()")
        rows = table.to_pylist()
        out_rows = self._infer_fn.infer(
            {
                "__THIS__": [SimpleNamespace(**row) for row in rows],
                "__STATE__": [self._state],
            }
        )
        out_dicts = [r.model_dump() for r in out_rows]
        return (
            pa.table({k: [r[k] for r in out_dicts] for k in out_dicts[0]})
            if out_dicts
            else pa.table({})
        )

    def _infer(self, row: dict[str, Any]) -> dict[str, Any]:
        """Single-row inference via InferFn."""
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before inference")
        out_rows = self._infer_fn.infer(
            {"__THIS__": [SimpleNamespace(**row)], "__STATE__": [self._state]}
        )
        return out_rows[0].model_dump()
