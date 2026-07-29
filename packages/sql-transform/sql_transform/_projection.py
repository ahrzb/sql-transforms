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

    def __init__(self, sql: str | Template) -> None:
        if isinstance(sql, Template):
            raise NotImplementedError("t-string templates are a later loop")
        self._marginalized = marginalize(sql)
        self._params: dict[str, pa.Table] | None = None

    @classmethod
    def from_file(cls, path: str) -> SQLProjection:
        """Build a projection from a file containing the SQL."""
        with open(path, encoding="utf-8") as f:
            return cls(f.read())

    def fit(
        self,
        table: pa.Table,
        /,
        this_model: type[BaseModel] | None = None,
    ) -> SQLProjection:
        """Materialize every params table over the training data; returns self.

        ``this_model`` is accepted for signature stability; the training table
        brings its own schema, so it is unused in this loop.
        """
        import duckdb

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
            if m.windows_sql is not None:
                # One execution of the original computation; params collapse
                # out of its materialized values (see Marginalized docstring).
                con.register(
                    "__CF_WINDOWS__", con.execute(m.windows_sql).to_arrow_table()
                )
            self._params = {
                spec.name: con.execute(spec.fit_sql).to_arrow_table()
                for spec in m.params
            }
        finally:
            con.close()
        return self

    @property
    def serving_sql(self) -> str:
        """The rewritten projection: params joins instead of aggregates."""
        return self._marginalized.serving_sql

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
