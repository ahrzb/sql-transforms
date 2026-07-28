"""SpecializedTransform — the SQLTransform surface served by the specializer."""

from __future__ import annotations

from string.templatelib import Template
from typing import Any

import datafusion
import pyarrow as pa
from confit import DuckDBInferFn
from pydantic import BaseModel

from sql_transform._compose import desugar_template, inline_references
from sql_transform._rewrite import rewrite_sql
from sql_transform._schema import synthesize_this_model
from sql_transform._sql import find_window_aggregates, parse_and_validate
from sql_transform._state import build_state_tables


class SpecializedTransform:
    """SQLTransform minus transformer refs, served by the SQL specializer.

    fit() runs the same state-extraction pipeline as SQLTransform (window
    aggregates freeze into per-partition static tables, template refs
    inline), then hands the rewritten SQL to the specializer: frontend ->
    binding-time analysis -> lowering -> native code, with the static
    tables baked in as prepare-time probe structures.

    infer()/infer_batch() take dicts or pydantic models directly and cross
    the boundary through the prepare-time-generated row marshaller.
    """

    def __init__(self, sql: str | Template) -> None:
        if isinstance(sql, Template):
            self._sql, self._refs = desugar_template(sql)
        else:
            self._sql, self._refs = sql, {}
        if any(r.is_transformer for r in self._refs.values()):
            raise ValueError(
                "SpecializedTransform does not support transformer refs; "
                "use SQLTransform"
            )
        self._fn: DuckDBInferFn | None = None

    @classmethod
    def from_file(cls, path: str) -> SpecializedTransform:
        with open(path) as f:
            return cls(f.read())

    def fit(
        self,
        table: pa.Table,
        /,
        this_model: type[BaseModel] | None = None,
    ) -> SpecializedTransform:
        this_model = this_model or synthesize_this_model(table.schema)
        tree = parse_and_validate(self._sql)

        ctx = datafusion.SessionContext()
        ctx.from_arrow(table, name="__THIS__")
        inline = inline_references(tree, self._refs, ctx, table)
        windows = find_window_aggregates(tree)
        own_state = build_state_tables(
            windows, ctx, "__THIS__", join_tables=inline.scoped_state
        )
        state_tables = {**inline.scoped_state, **own_state}
        rewritten = rewrite_sql(
            tree, windows, extra_marker_tables=tuple(inline.scoped_state)
        )
        self._fn = DuckDBInferFn(
            rewritten,
            row_tables={"__THIS__": this_model},
            static_tables=state_tables,
        )
        return self

    @property
    def backend(self) -> str:
        """Execution backend: "cranelift", "interpreter", or "constant"."""
        return self._fn_or_raise().backend

    @property
    def boundary(self) -> str:
        """Boundary path: "marshaller", "generic", or "constant"."""
        return self._fn_or_raise().boundary

    def infer(self, row: dict[str, Any] | BaseModel, /) -> BaseModel:
        """Single-row inference; returns the typed output model instance."""
        return self._fn_or_raise().infer_rows([row])[0]

    def infer_batch(self, rows: list[dict[str, Any] | BaseModel], /) -> list[BaseModel]:
        """Many-rows inference; returns typed output model instances."""
        return self._fn_or_raise().infer_rows(rows)

    def _fn_or_raise(self) -> DuckDBInferFn:
        if self._fn is None:
            raise RuntimeError("Must call fit() before inference")
        return self._fn
