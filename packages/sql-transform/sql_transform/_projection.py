"""SQLProjection — projections over ``__THIS__``, fit once, serve row-at-a-time.

The fit half works: window aggregates over ``__THIS__`` are marginalized into
materialized params tables plus a rewritten ``serving_sql`` (see
``_marginalize``). Transformers and author UDFs serve as scalar calls in that
SQL — ``transform`` registers them on the connection and runs; there is no
post-processing. The serving half (``infer``/``infer_batch`` through Confit)
is a later loop and still raises ``NotImplementedError``.
"""

from __future__ import annotations

from string.templatelib import Template
from typing import Any

import pyarrow as pa
from pydantic import BaseModel

from sql_transform._marginalize import MarginalizeError, marginalize
from sql_transform._udf import UDF, PythonTransform


def _feature_matrix(table: pa.Table, cols: list[str]):
    """Feature block as a 2D numpy array (float when possible, else object)."""
    import numpy as np

    raw = [table.column(c).to_pylist() for c in cols]
    try:
        mat = np.array(raw, dtype=float).T
    except TypeError, ValueError:
        mat = np.array(raw, dtype=object).T
    return mat


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
        transformers: dict[str, Any] | None = None,
    ) -> None:
        """``this_model`` declares the ``__THIS__`` schema (pydantic field
        names, in definition order — the model is authoritative). With it,
        unknown columns refuse here, stars/COLUMNS expand with modifiers at
        any level, and lateral aliases resolve by DuckDB's column-wins rule.
        Without it, marginalization is schema-free and the ambiguous cases
        refuse with a hint.

        ``transformers`` is the explicit registry for transformer windows
        (objects with fit/transform, called ``fn(...) OVER (...)``) and for
        author UDFs (objects with declared takes/returns, e.g. ``PythonUDF``,
        called ``fn(...)``); names not found there resolve from the caller's
        scope."""
        if isinstance(sql, Template):
            raise NotImplementedError("t-string templates are a later loop")
        self._columns = (
            list(this_model.model_fields) if this_model is not None else None
        )
        # Resolution: explicit registry first, then the caller's scope (the
        # `FROM df` replacement-scan idiom). Captured objects are snapshotted
        # into `self.transformers` at construction.
        import sys

        frame = sys._getframe(1)
        self.transformers: dict[str, Any] = {}

        def _resolve(name: str):
            obj = None
            if transformers is not None:
                obj = transformers.get(name)
            if obj is None and frame is not None:
                obj = frame.f_locals.get(name) or frame.f_globals.get(name)
            if obj is not None:
                self.transformers[name] = obj
            return obj

        self._marginalized = marginalize(sql, self._columns, _resolve)
        self._params: dict[str, pa.Table] | None = None
        self._udfs: dict[str, PythonTransform] | None = None

    @classmethod
    def from_file(cls, path: str) -> SQLProjection:
        """Build a projection from a file containing the SQL."""
        with open(path, encoding="utf-8") as f:
            return cls(f.read())

    def _register_author_udfs(self, con) -> None:
        for name in self._marginalized.scalar_udfs:
            self.transformers[name].register(con)

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
            self._register_author_udfs(con)
            # The fit plan is a topologically ordered DAG: run each step,
            # register its result under its name. Every intermediate is
            # inspectable; SQL steps are runnable by hand, fit steps produce
            # a params table (join keys + __cf_est) plus the fitted UDF.
            materialized: dict[str, pa.Table] = {}
            self._udfs = {}
            for step in m.plan:
                if step.kind == "fit":
                    params_table, udf = self._fit_step(
                        step, materialized[step.reads[0]]
                    )
                    materialized[step.name] = params_table
                    self._udfs[udf.name] = udf
                else:
                    materialized[step.name] = con.execute(step.sql).to_arrow_table()
                con.register(step.name, materialized[step.name])
            self._params = {spec.name: materialized[spec.name] for spec in m.params}
        finally:
            con.close()
        return self

    def _fit_step(self, step, table: pa.Table) -> tuple[pa.Table, PythonTransform]:
        """Group by the key columns, fit a clone of the transformer per group,
        and emit the params table linking each key tuple to its instance id.
        An unseen group at serving time misses the LEFT JOIN — NULL id, NULL
        output — so only observed groups get rows here."""
        if table.num_rows == 0:
            raise MarginalizeError(
                f"cannot fit transformer {step.transformer} on an empty training set"
            )
        proto = self.transformers[step.transformer]
        try:
            from sklearn.base import clone
        except ImportError:  # duck-typed objects without sklearn installed
            import copy as _copy

            def clone(o):  # type: ignore[no-redef]
                return _copy.deepcopy(o)

        import numpy as np

        m = self._marginalized
        (spec,) = [u for u in m.udfs if u.step == step.name]
        (params_spec,) = [p for p in m.params if p.name == step.name]
        feats = _feature_matrix(table, list(step.features))
        keyvals = [table.column(k).to_pylist() for k in step.keys]
        groups: dict[tuple, list[int]] = {}
        for i in range(table.num_rows):
            groups.setdefault(tuple(k[i] for k in keyvals), []).append(i)
        instances: dict[int, Any] = {}
        width = None
        key_rows: list[tuple] = []
        for est_id, (key, idx) in enumerate(groups.items()):
            est = clone(proto)
            est.fit(feats[idx])
            instances[est_id] = est
            key_rows.append(key)
            if width is None:
                probe = np.asarray(est.transform(feats[idx][:1]))
                width = probe.shape[1] if probe.ndim > 1 else 1
        cols: dict[str, pa.Array] = {}
        for pos, colname in enumerate(params_spec.keys):
            cols[colname] = pa.array(
                [k[pos] for k in key_rows],
                type=table.column(step.keys[pos]).type,
            )
        cols["__cf_est"] = pa.array(range(len(key_rows)), type=pa.int64())
        udf = PythonTransform(
            name=spec.name,
            instances=instances,
            takes=("f64",) * len(step.features),
            returns=("f64",) * width,
        )
        return pa.table(cols), udf

    def transform(self, table: pa.Table, /) -> pa.Table:
        """Batch apply: register the params tables and every UDF (author UDFs
        and fitted transformers), run ``serving_sql``, done. Row-at-a-time
        serving stays with Confit."""
        import duckdb

        if self._params is None or self._udfs is None:
            raise MarginalizeError("not fitted: call fit(table) first")
        if self._columns is not None:
            table = table.select(self._columns)
        m = self._marginalized
        con = duckdb.connect()
        try:
            con.execute("SET threads = 1")
            con.register("__THIS__", table)
            for name, params_table in self._params.items():
                con.register(name, params_table)
            self._register_author_udfs(con)
            for udf in self._udfs.values():
                udf.register(con)
            return con.execute(m.serving_sql).to_arrow_table()
        finally:
            con.close()

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
    def udfs(self) -> dict[str, UDF]:
        """Every UDF ``serving_sql`` calls, by SQL name: author UDFs plus the
        fitted transformers (``__cf_tf{j}``). This dict plus ``params`` plus
        ``serving_sql`` is the complete serving artifact."""
        if self._udfs is None:
            raise MarginalizeError("not fitted: call fit(table) first")
        out: dict[str, UDF] = {
            name: self.transformers[name] for name in self._marginalized.scalar_udfs
        }
        out.update(self._udfs)
        return out

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
