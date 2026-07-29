"""SQLProjection — projections over ``__THIS__``, fit once, serve row-at-a-time.

The fit half works: window aggregates over ``__THIS__`` are marginalized into
materialized params tables plus a rewritten ``serving_sql`` (see
``_marginalize``). The serving half (``infer``/``infer_batch`` through Confit)
is a later loop and still raises ``NotImplementedError``.
"""

from __future__ import annotations

from string.templatelib import Template
from typing import Any

import pyarrow as pa
from pydantic import BaseModel

from sql_transform._marginalize import MarginalizeError, marginalize

_TODO = "SQLProjection serving is a later loop; {} has no implementation yet"


class SQLProjection:
    """A SQL projection: fit freezes its window aggregates into params tables.

    >>> p = SQLProjection(
    ...     "SELECT (age - avg(age) OVER (PARTITION BY country)) AS d FROM __THIS__"
    ... ).fit(train)
    >>> p.serving_sql   # the rewritten projection, joins instead of aggregates
    >>> p.params        # {"__CF_PARAMS_0__": <pyarrow.Table>}
    """

    def __init__(
        self,
        sql: str | Template,
        /,
        this_model: type[BaseModel] | None = None,
    ) -> None:
        """``this_model`` declares the ``__THIS__`` schema (pydantic field
        names, in definition order — the model is authoritative). With it,
        unknown columns refuse here, stars/COLUMNS expand with modifiers at
        any level, and lateral aliases resolve by DuckDB's column-wins rule.
        Without it, marginalization is schema-free and the ambiguous cases
        refuse with a hint."""
        if isinstance(sql, Template):
            raise NotImplementedError("t-string templates are a later loop")
        self._columns = (
            list(this_model.model_fields) if this_model is not None else None
        )
        self._marginalized = marginalize(sql, self._columns)
        self._params: dict[str, pa.Table] | None = None

    @classmethod
    def from_file(cls, path: str) -> SQLProjection:
        """Build a projection from a file containing the SQL."""
        with open(path, encoding="utf-8") as f:
            return cls(f.read())

    def fit(self, table: pa.Table, /) -> SQLProjection:
        """Materialize every params table over the training data; returns self.

        When a ``this_model`` was declared, it is authoritative: the table is
        canonicalized to the model's columns in model order (extra table
        columns drop; missing ones refuse by name).
        """
        import duckdb

        if self._columns is not None:
            missing = [c for c in self._columns if c not in table.column_names]
            if missing:
                raise MarginalizeError(
                    f"training table is missing model column {missing[0]}"
                )
            table = table.select(self._columns)
        m = self._marginalized
        con = duckdb.connect()
        try:
            # DuckDB's parallel window aggregation accumulates floats in a
            # schedule-dependent order (measured: 1/500 fuzz cases drift an
            # ulp). Single-threaded fit makes params deterministic and
            # machine-reproducible.
            # ponytail: threads=1; a parallel fit needs a determinism story.
            con.execute("SET threads = 1")
            con.register("__THIS__", table)
            # The fit plan is a topologically ordered DAG: run each step,
            # register its result under its name. Every intermediate is
            # inspectable and every step is plain SQL runnable by hand.
            materialized: dict[str, pa.Table] = {}
            for step in m.plan:
                materialized[step.name] = con.execute(step.sql).to_arrow_table()
                con.register(step.name, materialized[step.name])
            self._params = {spec.name: materialized[spec.name] for spec in m.params}
        finally:
            con.close()
        return self

    @property
    def serving_sql(self) -> str:
        """The rewritten projection: params joins instead of aggregates."""
        return self._marginalized.serving_sql

    @property
    def plan(self):
        """The fit plan: an ordered tuple of named FitSteps (debuggable SQL)."""
        return self._marginalized.plan

    @property
    def params(self) -> dict[str, pa.Table]:
        """The materialized params tables, by name; fitted only."""
        if self._params is None:
            raise MarginalizeError("not fitted: call fit(table) first")
        return dict(self._params)

    @property
    def backend(self) -> str:
        """Execution backend: "cranelift", "interpreter", or "constant"."""
        raise NotImplementedError(_TODO.format("backend"))

    @property
    def boundary(self) -> str:
        """Boundary path: "marshaller", "generic", or "constant"."""
        raise NotImplementedError(_TODO.format("boundary"))

    def infer(self, row: dict[str, Any] | BaseModel, /) -> BaseModel:
        """Single-row inference; returns the typed output model instance."""
        raise NotImplementedError(_TODO.format("infer"))

    def infer_batch(self, rows: list[dict[str, Any] | BaseModel], /) -> list[BaseModel]:
        """Many-rows inference; returns typed output model instances."""
        raise NotImplementedError(_TODO.format("infer_batch"))
