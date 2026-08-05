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

from sql_transform._marginalize import (
    MarginalizeError,
    marginalize,
)
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


def _row_ordered_sql(con: Any, sql: str) -> str:
    """The serving text with the input's ``__cf_row`` threaded through every
    spine SELECT (the statement node and its CTEs — never expression
    subqueries) and a final ORDER BY on it. Every level is row-preserving
    (the surface admits no GROUP BY), so the extra item is always lawful.
    Node shapes are harvested from a serialized template, never hand-built."""
    import json

    (ser,) = con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()
    ast = json.loads(ser)
    (tser,) = con.execute(
        "SELECT json_serialize_sql("
        "'SELECT __cf_row AS __cf_row FROM t ORDER BY __cf_row')"
    ).fetchone()
    template = json.loads(tser)["statements"][0]["node"]
    item, modifiers = template["select_list"][0], template["modifiers"]
    node = ast["statements"][0]["node"]
    spine = [node] + [
        e["value"]["query"]["node"] for e in node.get("cte_map", {}).get("map", [])
    ]
    for s in spine:
        s["select_list"] = [*s["select_list"], item]
    node["modifiers"] = [*node.get("modifiers", []), *modifiers]
    (out,) = con.execute(
        "SELECT json_deserialize_sql(?::JSON)", [json.dumps(ast)]
    ).fetchone()
    return out


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

        ``transformers`` is the explicit registry for transformers (objects
        with fit/transform — bare ``tfm(bundle).field`` is a global
        fit-transform; other fit scopes spell the split,
        ``tfm_transform(tfm_fit(bundle) OVER (...), bundle).field``; a
        registered ``x`` reserves ``x_fit``/``x_transform``) and for author
        UDFs (objects with declared takes/returns, e.g. ``PythonUDF``,
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
        self._serving_sql: str | None = None  # finalized at fit (wide items)

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
                    # ONE UDF per step (TASK-63): whole-value calls and
                    # field reads all serve through it. Every addressed
                    # field must exist in the fitted output — checked here,
                    # at fit, not construction: T is learned (DRAFT-24, the
                    # P7 carve-out).
                    self._udfs[udf.name] = udf
                    for spec in m.udfs:
                        if (
                            spec.step == step.name
                            and spec.field is not None
                            # ASCII-case-insensitive, like both engines'
                            # struct reads (case COLLISIONS refuse earlier).
                            and spec.field.lower()
                            not in {n.lower() for n in udf.return_names}
                        ):
                            raise MarginalizeError(
                                f"transformer {step.transformer} has no output"
                                f" field {spec.field!r}; it fits to"
                                f" {list(udf.return_names)}"
                            )
                else:
                    materialized[step.name] = con.execute(step.sql).to_arrow_table()
                con.register(step.name, materialized[step.name])
            self._params = {spec.name: materialized[spec.name] for spec in m.params}
        finally:
            con.close()
        # Struct-valued calls: the serving text is final at marginalize —
        # every transformer mention is a field read over the one call, at
        # every width. Nothing left to rewrite at fit.
        self._serving_sql = m.serving_sql
        self._row_model = _model_from_arrow(table.schema)
        self._fn = None  # refit invalidates the prepared serving function
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
        specs = [u for u in m.udfs if u.step == step.name]
        (params_spec,) = [p for p in m.params if p.name == step.name]
        base_name = specs[0].name  # every spec of a step shares the one UDF
        # Typed BEFORE fitting: a nested feature column dies here by name,
        # never as a raw error inside est.fit (review round 2026-08-05).
        takes = tuple(
            _engine_ty(table.column(c).type, n)
            for c, n in zip(step.features, step.feature_names, strict=True)
        )
        feats = _feature_matrix(table, list(step.features))
        keyvals = [table.column(k).to_pylist() for k in step.keys]
        # FILTER on the fit call: only predicate-TRUE rows (the level table
        # holds CAST(pred AS BOOLEAN), so SQL's three-valued logic already
        # happened) enter the fit; a group with no passing rows gets no
        # params row — an unseen group, NULL at serving (P14).
        mask = table.column(step.filter_col).to_pylist() if step.filter_col else None
        groups: dict[tuple, list[int]] = {}
        for i in range(table.num_rows):
            if mask is not None and mask[i] is not True:
                continue
            groups.setdefault(tuple(k[i] for k in keyvals), []).append(i)
        if not groups:
            raise MarginalizeError(
                f"FILTER on transformer {step.transformer} left no training"
                " rows — the fitted output shape is unlearnable"
            )
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
            # Field matching is ASCII-case-insensitive on both engines
            # (DuckDB struct keys; confit lane binding) — colliding fitted
            # names would serve silently wrong values.
            if len({n.lower() for n in names}) != len(names):
                raise MarginalizeError(
                    f"transformer {step.transformer} fits to case-colliding"
                    f" output names {list(names)} — field matching is"
                    " case-insensitive"
                )
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
            takes=takes,
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
        con = duckdb.connect()
        try:
            con.execute("SET threads = 1")
            # SQL results are unordered, and the params LEFT JOIN really
            # does emit unmatched (unseen-group) probe rows last — restore
            # input order explicitly (2026-08-05 review round).
            table = table.append_column(
                "__cf_row", pa.array(range(table.num_rows), type=pa.int64())
            )
            con.register("__THIS__", table)
            for name, params_table in self._params.items():
                con.register(name, params_table)
            self._register_author_udfs(con)
            for udf in self._udfs.values():
                udf.register(con)
            out = con.execute(_row_ordered_sql(con, self.serving_sql)).to_arrow_table()
            # Only a schema-free * can smuggle a _-named column this far
            # (authored ones refuse at construction); privates never cross
            # the output boundary. The row path cannot express one at all —
            # the row model drops _-leading fields.
            leaked = [
                c
                for c in out.column_names
                if c.startswith("_") and not c.startswith("__cf_")
            ]
            if leaked:
                raise MarginalizeError(
                    f"column {leaked[0]} crossed the output boundary via *"
                    " (output fields starting with _ are private) — declare a"
                    " this_model or rename it"
                )
            return out.select(
                [i for i, c in enumerate(out.column_names) if c != "__cf_row"]
            )
        finally:
            con.close()

    @property
    def serving_sql(self) -> str:
        """The rewritten projection: params joins instead of aggregates.

        Final after fit: a bare width-k transformer item cannot be spelled
        before its field names are learned, so fit expands it into one
        aliased lane call per field."""
        if self._serving_sql is not None:
            return self._serving_sql
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
                self.serving_sql,
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
