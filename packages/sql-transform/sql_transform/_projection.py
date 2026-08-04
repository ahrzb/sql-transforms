"""SQLProjection — projections over ``__THIS__``, fit once, serve row-at-a-time.

The fit half: window aggregates over ``__THIS__`` are marginalized into
materialized params tables plus a rewritten ``serving_sql`` (see
``_marginalize``); transformers fit per group into ``PythonTransform`` UDFs.
The serving half is two bindings of the same artifact: ``transform`` runs
``serving_sql`` through DuckDB with the UDFs registered (batch, the oracle),
and ``infer``/``infer_batch`` run it through Confit's ``DuckDBInferFn`` with
the same UDF objects passed as ``udfs=`` (row-at-a-time, bit-exact with the
DuckDB path by Confit's contract).
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


def _engine_ty(t: pa.DataType, where: str) -> str:
    """An arrow column type in the engine's vocabulary; refuses by name for
    anything a UDF argument cannot carry."""
    if pa.types.is_floating(t):
        return "f64"
    if pa.types.is_boolean(t):
        return "i1"
    if pa.types.is_integer(t):
        return "i64"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "str"
    raise MarginalizeError(
        f"transformer feature {where} has type {t}, which no UDF argument"
        " can carry (i1/i64/f64/str)"
    )


def _model_from_arrow(schema: pa.Schema) -> type[BaseModel]:
    """The serving row model, derived from the training table's schema (real
    types, guaranteed — a declared ``this_model`` may carry ``object``
    fields). Unmappable types become opaque ``object`` fields: Confit
    accepts those unless the SQL references them."""
    import pydantic

    fields: dict[str, Any] = {}
    for f in schema:
        t = f.type
        if pa.types.is_floating(t):
            p: type = float
        elif pa.types.is_integer(t):
            p = int
        elif pa.types.is_boolean(t):
            p = bool
        elif pa.types.is_string(t) or pa.types.is_large_string(t):
            p = str
        else:
            p = object
        fields[f.name] = (p | None if p is not object else Any, None)
    return pydantic.create_model("Row", **fields)


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
        self._row_model: type[BaseModel] | None = None
        self._fn: Any = None  # confit.DuckDBInferFn, built lazily post-fit

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
                    # The base UDF is registered only when the query calls it
                    # whole; addressed fields serve through their lane UDFs.
                    if any(u.name == udf.name and u.field is None for u in m.udfs):
                        self._udfs[udf.name] = udf
                    self._udfs.update(self._lane_udfs(step, udf))
                else:
                    materialized[step.name] = con.execute(step.sql).to_arrow_table()
                con.register(step.name, materialized[step.name])
            self._params = {spec.name: materialized[spec.name] for spec in m.params}
        finally:
            con.close()
        self._row_model = _model_from_arrow(table.schema)
        self._fn = None  # refit invalidates the prepared serving function
        return self

    def _lane_udfs(self, step, base: PythonTransform) -> dict[str, Any]:
        """The width-1 lane UDFs for every field this query addresses on
        ``base``. A requested field absent from the fitted output refuses
        here — at fit, not at construction: T is learned, so its names are
        not knowable earlier (DRAFT-24, the P7 carve-out)."""
        from sql_transform._udf import TransformLane, UDFError

        out: dict[str, Any] = {}
        for spec in self._marginalized.udfs:
            if spec.step != step.name or spec.field is None:
                continue
            try:
                lane = base.lane_of(spec.field)
            except UDFError:
                raise MarginalizeError(
                    f"transformer {step.transformer} has no output field"
                    f" {spec.field!r}; it fits to"
                    f" {list(base.return_names)}"
                ) from None
            out[spec.name] = TransformLane(name=spec.name, source=base, lane=lane)
        return out

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

        def clone(o):
            """sklearn's clone when the object is an estimator; a deepcopy
            otherwise — duck-typed transformers are first-class, and
            sklearn's clone raises TypeError (not ImportError) on them."""
            try:
                from sklearn.base import clone as _sk_clone

                return _sk_clone(o)
            except ImportError, TypeError:
                import copy as _copy

                return _copy.deepcopy(o)

        import numpy as np

        from sql_transform._udf import output_names

        m = self._marginalized
        # The whole-value spec if the query uses one; otherwise any spec of
        # this step names the base UDF (lane UDFs hang off it).
        specs = [u for u in m.udfs if u.step == step.name]
        (params_spec,) = [p for p in m.params if p.name == step.name]
        base_name = next(
            (u.name for u in specs if u.field is None),
            specs[0].name.rsplit("_g", 1)[0],  # the lane UDFs' shared source
        )
        feats = _feature_matrix(table, list(step.features))
        keyvals = [table.column(k).to_pylist() for k in step.keys]
        groups: dict[tuple, list[int]] = {}
        for i in range(table.num_rows):
            groups.setdefault(tuple(k[i] for k in keyvals), []).append(i)
        instances: dict[int, Any] = {}
        shape: tuple[str, ...] | None = None
        key_rows: list[tuple] = []
        for est_id, (key, idx) in enumerate(groups.items()):
            est = clone(proto)
            est.fit(feats[idx])
            instances[est_id] = est
            key_rows.append(key)
            probe = np.asarray(est.transform(feats[idx][:1]))
            width = probe.shape[1] if probe.ndim > 1 else 1
            names = output_names(est, step.feature_names, width, step.transformer)
            # Every group must fit to the SAME output struct — a codomain
            # that varies per group is not a function type (an encoder whose
            # categories differ per partition hits this).
            if shape is None:
                shape = names
            elif names != shape:
                raise MarginalizeError(
                    f"transformer {step.transformer} fits to different output"
                    f" shapes per group: {list(shape)} vs {list(names)}"
                    f" (group {key})"
                )
        cols: dict[str, pa.Array] = {}
        for pos, colname in enumerate(params_spec.keys):
            cols[colname] = pa.array(
                [k[pos] for k in key_rows],
                type=table.column(step.keys[pos]).type,
            )
        cols["__cf_est"] = pa.array(range(len(key_rows)), type=pa.int64())
        assert shape is not None  # groups is non-empty: the table has rows
        udf = PythonTransform(
            name=base_name,
            instances=instances,
            # S's field types come from the level table — the real types of
            # the bundle expressions, so a string feature stays a string.
            takes=tuple(
                _engine_ty(table.column(c).type, n)
                for c, n in zip(step.features, step.feature_names, strict=True)
            ),
            returns=("f64",) * len(shape),
            take_names=tuple(step.feature_names),
            return_names=shape,
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

    def _serving_fn(self):
        """The Confit binding of the fitted artifact, prepared once: the same
        ``serving_sql``, params tables, and UDF objects the DuckDB path uses —
        Confit's contract makes the two bit-exact or refuses by name."""
        if self._params is None or self._udfs is None or self._row_model is None:
            raise MarginalizeError("not fitted: call fit(table) first")
        if self._fn is None:
            from confit import DuckDBInferFn

            m = self._marginalized
            udfs = [self.transformers[n] for n in m.scalar_udfs]
            udfs += list(self._udfs.values())
            self._fn = DuckDBInferFn(
                m.serving_sql,
                row_tables={"__THIS__": self._row_model},
                static_tables=dict(self._params),
                udfs=udfs,
                shape="map",
            )
        return self._fn

    @property
    def backend(self) -> str:
        """Execution backend: "cranelift", "interpreter", or "constant"."""
        return self._serving_fn().backend

    @property
    def boundary(self) -> str:
        """Boundary path: "marshaller", "generic", or "constant"."""
        return self._serving_fn().boundary

    @property
    def output_model(self) -> type[BaseModel]:
        """The typed output row model ``infer`` returns instances of."""
        return self._serving_fn().output_model

    def infer(self, row: dict[str, Any] | BaseModel, /) -> BaseModel:
        """Single-row inference; returns the typed output model instance."""
        (out,) = self._serving_fn().infer_rows([row])  # shape="map": exactly one
        return out

    def infer_batch(self, rows: list[dict[str, Any] | BaseModel], /) -> list[BaseModel]:
        """Many-rows inference; returns typed output model instances."""
        return self._serving_fn().infer_rows(rows)
