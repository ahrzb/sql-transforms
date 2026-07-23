"""SQLTransform — sklearn-compatible SQL-based feature transforms."""

from __future__ import annotations

from string.templatelib import Template
from types import SimpleNamespace
from typing import Any, Literal

import datafusion
import numpy as np
import pyarrow as pa
from pydantic import BaseModel

from sql_transform._batch import run_batch
from sql_transform._compose import desugar_template, inline_references
from sql_transform._interpreter import InferFn
from sql_transform._rewrite import rewrite_sql
from sql_transform._schema import synthesize_this_model
from sql_transform._sql import find_window_aggregates, parse_and_validate
from sql_transform._state import build_state_tables
from sql_transform._transformer_ref import resolve_transformer_refs

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

    def __init__(
        self,
        sql: str | Template,
        *,
        output: Literal["records", "dense"] = "records",
    ) -> None:
        if output not in ("records", "dense"):
            raise ValueError(f"output must be 'records' or 'dense', got {output!r}")
        self._output = output
        if isinstance(sql, Template):
            self._sql, self._refs = desugar_template(sql)
        else:
            self._sql, self._refs = sql, {}
        self._state_tables: dict[str, pa.Table] | None = None
        self._rewritten_sql: str | None = None
        self._infer_fn: InferFn | None = None
        self._udf_specs: dict[str, tuple] = {}

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

        ctx = datafusion.SessionContext()
        ctx.from_arrow(table, name="__THIS__")

        sqlt_refs = {n: r for n, r in self._refs.items() if not r.is_transformer}
        tfm_refs = {n: r.transform for n, r in self._refs.items() if r.is_transformer}
        self._udf_specs = resolve_transformer_refs(tree, tfm_refs, table)

        inline = inline_references(tree, sqlt_refs, ctx, table)
        windows = find_window_aggregates(tree)

        own_state = build_state_tables(
            windows, ctx, "__THIS__", join_tables=inline.scoped_state
        )
        self._state_tables = {**inline.scoped_state, **own_state}
        self._rewritten_sql = rewrite_sql(
            tree, windows, extra_marker_tables=tuple(inline.scoped_state)
        )
        self._infer_fn = InferFn(
            self._rewritten_sql,
            row_tables={"__THIS__": this_model},
            static_tables=self._state_tables,
            transformers={
                n: (obj, out_s) for n, (obj, in_s, out_s) in self._udf_specs.items()
            },
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
        out = run_batch(
            self._rewritten_sql, table, self._state_tables, self._udf_specs
        )
        if self._output == "dense":
            return _table_to_dense(out)
        return out

    def infer(
        self, row: dict[str, Any] | BaseModel, /
    ) -> BaseModel | np.ndarray:
        """Single-row inference through the Rust InferFn against the frozen
        state. Accepts a dict or a Pydantic model; returns the typed output
        model instance (records mode) or a float64 (k,) row (dense mode)."""
        return self.infer_batch([row])[0]

    def infer_batch(
        self, rows: list[dict[str, Any] | BaseModel], /
    ) -> list[BaseModel] | np.ndarray:
        """Many-rows inference through the Rust InferFn against the frozen
        state. Accepts dicts and/or Pydantic models; returns a list of typed
        output model instances (records mode) or a float64 (n,k) matrix
        (dense mode, NULL -> NaN)."""
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before inference")
        this_rows = [_to_namespace(row) for row in rows]
        models = self._infer_fn.infer({"__THIS__": this_rows})
        if self._output == "dense":
            # ponytail: goes through pydantic records; a direct columnar path
            # from the Rust engine is the upgrade if this shows up in profiles.
            nan = float("nan")
            data = [
                [nan if v is None else v for v in m.model_dump().values()]
                for m in models
            ]
            k = len(models[0].model_dump()) if models else 0
            return np.asarray(data, dtype=np.float64).reshape(len(models), k)
        return models


def _table_to_dense(table: pa.Table) -> np.ndarray:
    """pyarrow Table -> dense float64 (n,k) matrix, NULL -> NaN."""
    cols = [
        table.column(i).cast(pa.float64()).to_numpy(zero_copy_only=False)
        for i in range(table.num_columns)
    ]
    return np.column_stack(cols) if cols else np.empty((table.num_rows, 0))
