"""A foreign transform: the ``(fit, transform)`` pair, supplied directly.

In SQL the pair splits into two DuckDB functions joined by θ, an opaque handle
into a registry of fitted instances. An SQL leaf gives an inspectable params
table; a fitted RandomForest gives a pointer.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from sql_transform._udf import (
    _ARROW,
    PythonTransform,
    UDFError,
    _as_feature,
    output_names,
)
from sql_transform.model._ast import Connection
from sql_transform.model._errors import TransformError

THETA_SQL = "STRUCT(type VARCHAR, id BIGINT)"


THETA_ARROW = pa.struct([("type", pa.string()), ("id", pa.int64())])


def _struct_sql(fields: tuple[str, ...]) -> str:
    for name in fields:
        if not name.isidentifier():
            raise TransformError(f"{name!r} is not a usable struct field name")
    return "STRUCT(" + ", ".join(f"{f} DOUBLE" for f in fields) + ")"


class _Registry:
    """The fitted instances a run mints, and the first error it hit.

    Ids come from a monotone counter under a lock, never from
    ``len(instances)``. Measured with the length form, fitting two categories:
    both θ rows carried ``id 0`` and only one instance was stored, so every
    row of one category was served by the other's estimator — silently, with
    plausible numbers. DuckDB fits groups on several threads, and the window
    is not one bytecode: ``iid = len(instances)`` is read, ``fit()`` runs, and
    only then is the instance stored, so the whole fit sits between the read
    and the write. ``ids_are_unique_under_concurrency`` reproduces exactly
    that shape.

    Nothing here leans on the GIL, and it must not: 3.14 supports
    free-threaded builds, where every window widens and a dict update is no
    longer atomic either.

    The lock also carries the first exception out, because DuckDB rewraps a
    Python exception as ``InvalidInputException`` and a refusal has to keep
    its name.

    ponytail: one lock for the whole registry. Per-instance locks only if fit
    ever becomes a throughput problem, which it is not — fit runs once.
    """

    def __init__(self, instances: dict[int, Any] | None = None) -> None:
        self.instances = dict(instances or {})
        self.error: Exception | None = None
        self._next = max(self.instances, default=-1) + 1
        self._lock = threading.Lock()

    def add(self, instance: Any) -> int:
        with self._lock:
            iid = self._next
            self._next += 1
            self.instances[iid] = instance
        return iid

    def keep(self, exc: Exception) -> None:
        """Remember the first error, without raising it."""
        with self._lock:
            if self.error is None:
                self.error = exc

    def fail(self, exc: Exception) -> None:
        with self._lock:
            if self.error is None:
                self.error = exc
        raise exc


def _execute(con: Connection, sql: str, registry: _Registry) -> pa.Table:
    """Run ``sql``, letting a foreign transform's own refusal out by name."""
    try:
        # to_arrow_table, never .arrow(): see _duckdb_arrow_test.py — the
        # reader .arrow() returns deadlocks when registered back.
        return con.execute(sql).to_arrow_table()
    except Exception as exc:
        if registry.error is not None:
            raise registry.error from exc
        raise


@dataclass(slots=True)
class _Estimator:
    """One authored raw fit scope, with runtime state private to each fit.

    Arrow supplies the actual bundle declaration. Fit and the unchanged
    PythonTransform then use the same feature conversion, including numeric
    NULLs as NaN and strings that stay strings.
    """

    prototype: Any
    feature_names: tuple[str, ...]
    fit_name: str
    udf_name: str
    _takes: pa.Schema | None = field(default=None, init=False)
    _returns: tuple[str, ...] | None = field(default=None, init=False)
    _instances: dict[int, Any] = field(default_factory=dict, init=False)

    def fresh(self) -> "_Estimator":
        return _Estimator(
            self.prototype, self.feature_names, self.fit_name, self.udf_name
        )

    def _fit_batch(self, groups: Any, registry: _Registry) -> pa.Array:
        import copy  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        try:
            kind = groups.type
            if not (pa.types.is_list(kind) or pa.types.is_large_list(kind)):
                raise TransformError(f"{self.fit_name}: fit requires a list of bundles")
            bundle = kind.value_type
            if not pa.types.is_struct(bundle):
                raise TransformError(
                    f"{self.fit_name}: fit requires a named struct bundle"
                )
            names = tuple(f.name for f in bundle)
            if len({n.lower() for n in names}) != len(names):
                raise TransformError(
                    f"{self.fit_name}: case-colliding feature names {list(names)}"
                )
            if names != self.feature_names:
                raise TransformError(
                    f"{self.fit_name}: expected feature fields "
                    f"{list(self.feature_names)}, received {list(names)}"
                )
            codes: list[str] = []
            for feature in bundle:
                ty = feature.type
                if pa.types.is_integer(ty):
                    code = "i64"
                elif pa.types.is_floating(ty):
                    code = "f64"
                elif pa.types.is_boolean(ty):
                    code = "i1"
                elif pa.types.is_string(ty) or pa.types.is_large_string(ty):
                    code = "str"
                else:
                    raise TransformError(
                        f"{self.fit_name}: feature {feature.name!r} type {ty} "
                        "is unsupported; use integer, floating, boolean or string"
                    )
                codes.append(code)
            takes = pa.schema(
                [(name, _ARROW[code]) for name, code in zip(names, codes, strict=True)]
            )
            if self._takes is not None and not takes.equals(self._takes):
                raise TransformError(
                    f"{self.fit_name}: different feature schemas within one fit scope"
                )
            self._takes = takes
            ids: list[int | None] = []
            for group in groups.to_pylist():
                if not group:
                    ids.append(None)
                    continue
                matrix = np.asarray(
                    [
                        [
                            _as_feature(row[name], code)
                            for name, code in zip(names, codes, strict=True)
                        ]
                        for row in group
                    ],
                    dtype=(
                        object if any(c in ("str", "i1") for c in codes) else np.float64
                    ),
                )
                try:
                    from sklearn.base import clone  # noqa: PLC0415

                    est = clone(self.prototype)
                except (ImportError, TypeError):
                    est = copy.deepcopy(self.prototype)
                est.fit(matrix)
                try:
                    probe = np.asarray(est.transform(matrix[:1]))
                except (TypeError, ValueError) as exc:
                    raise TransformError(
                        f"{self.fit_name}: unsupported transform output shape"
                    ) from exc
                if probe.ndim == 1 and probe.shape == (1,):
                    width = 1
                elif probe.ndim == 2 and probe.shape[0] == 1 and probe.shape[1] > 0:
                    width = probe.shape[1]
                else:
                    raise TransformError(
                        f"{self.fit_name}: unsupported transform output shape "
                        f"{probe.shape}; expected one row of scalar values"
                    )
                try:
                    for value in probe.flat:
                        float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise TransformError(
                        f"{self.fit_name}: transform output must contain DOUBLE values"
                    ) from exc
                try:
                    returns = output_names(est, names, width, self.fit_name)
                except UDFError as exc:
                    raise TransformError(str(exc)) from exc
                if len({n.lower() for n in returns}) != len(returns):
                    raise TransformError(
                        f"{self.fit_name}: case-colliding output names {list(returns)}"
                    )
                if self._returns is not None and returns != self._returns:
                    raise TransformError(
                        f"{self.fit_name}: different output shapes per group: "
                        f"{list(self._returns)} vs {list(returns)}"
                    )
                self._returns = returns
                iid = registry.add(est)
                self._instances[iid] = est
                ids.append(iid)
            return pa.array(ids, type=pa.int64())
        except Exception as exc:
            registry.keep(exc)
            raise

    def register(
        self, con: Connection, leased_fit_name: str, registry: _Registry
    ) -> None:
        con.create_function(
            leased_fit_name,
            lambda groups: self._fit_batch(groups, registry),
            parameters=None,
            return_type="BIGINT",
            type="arrow",
            null_handling="special",
            side_effects=True,
        )

    def publish(self) -> PythonTransform:
        if not self._instances or self._takes is None or self._returns is None:
            raise TransformError(
                f"{self.fit_name}: cannot fit on empty or entirely filtered training "
                "data; the fitted output shape is unlearnable"
            )
        return PythonTransform(
            self.udf_name,
            self._instances,
            self._takes,
            pa.struct([(name, pa.float64()) for name in self._returns]),
        )


@dataclass(slots=True)
class Transform:
    """A foreign transform: the ``(fit, transform)`` pair, supplied directly.

    ``fit(F) -> instance`` and ``transform(instance, T) -> R``, both over
    relations. Fit receives a complete aggregate group; transform receives
    one instance's subset of a single Arrow invocation, not the whole request.

    In SQL the pair splits: ``x_fit`` is the UDAF half and ``x_transform`` the
    UDF half, joined by θ, an opaque ``Struct<type, id>`` handle into a
    registry of fitted instances. An SQL leaf gives an inspectable, shippable
    params table; a fitted RandomForest gives a pointer.

    ``takes``/``returns`` name the input and output struct fields, and are
    author-declared rather than inferred: DuckDB has no ``ANY`` type, so the
    shapes must be concrete before the functions can be registered at all. The
    declaration is authoritative — a transform whose output width disagrees
    refuses rather than mislabelling lanes.

    Everything is DOUBLE. Widening the vocabulary is a later problem; nothing
    in the design turns on it.
    """

    fit: Callable[[pa.Table], Any]
    transform: Callable[[Any, pa.Table], pa.Table]
    takes: tuple[str, ...]
    returns: tuple[str, ...]

    def __post_init__(self) -> None:
        _struct_sql(self.takes)  # a bad field name refuses here, not at fit
        _struct_sql(self.returns)

    # -- the two SQL halves ----------------------------------------------------

    def _fit_batch(self, groups: Any, stem: str, registry: _Registry) -> pa.Array:
        thetas = []
        for group in groups.to_pylist():
            if group is None:
                thetas.append(None)
                continue
            relation = pa.table(
                {
                    field: pa.array([row[field] for row in group], pa.float64())
                    for field in self.takes
                }
            )
            # DuckDB rewraps a Python exception from a UDF, so a leaf's own
            # named refusal reaches fit() unrecognisable. The registry is
            # where the first real error is kept — put it there before it is
            # buried.
            try:
                fitted = self.fit(relation)
            except Exception as exc:
                registry.keep(exc)
                raise
            thetas.append({"type": stem, "id": registry.add(fitted)})
        return pa.array(thetas, type=THETA_ARROW)

    def _transform_batch(
        self, theta: Any, features: Any, stem: str, registry: _Registry
    ) -> pa.Array:
        thetas, feats = theta.to_pylist(), features.to_pylist()
        out: list[Any] = [None] * len(thetas)
        rows_by_instance: dict[int, list[int]] = {}
        for i, handle in enumerate(thetas):
            # P14, the one NULL story: a NULL θ is a LEFT JOIN miss, which is
            # an unseen group. The row stays, its output is NULL.
            if handle is not None:
                rows_by_instance.setdefault(handle["id"], []).append(i)

        for iid, positions in rows_by_instance.items():
            if iid not in registry.instances:
                registry.fail(
                    TransformError(
                        f"{stem}: θ id {iid} is not in the fitted instances — "
                        "the params table and the instances are from different fits"
                    )
                )
            relation = pa.table(
                {
                    field: pa.array([feats[i][field] for i in positions], pa.float64())
                    for field in self.takes
                }
            )
            try:
                produced = self.transform(registry.instances[iid], relation)
            except Exception as exc:
                registry.keep(exc)
                raise
            if tuple(produced.column_names) != self.returns:
                registry.fail(
                    TransformError(
                        f"{stem}: declared width {self.returns} but produced "
                        f"{tuple(produced.column_names)}"
                    )
                )
            values = produced.to_pylist()
            for position, value in zip(positions, values, strict=True):
                out[position] = value
        return pa.array(out, type=pa.struct([(f, pa.float64()) for f in self.returns]))

    def register(self, con: Connection, stem: str, registry: _Registry) -> None:
        """Bind both halves to a connection. Both are always registered: a
        fit-only subtree may transform, and ``x_fit`` over ``__THIS__`` is
        legal and means refit on the batch you were handed."""
        struct_in = _struct_sql(self.takes)
        con.create_function(
            f"{stem}_fit",
            lambda groups: self._fit_batch(groups, stem, registry),
            [f"{struct_in}[]"],
            THETA_SQL,
            type="arrow",
            null_handling="special",
        )
        con.create_function(
            f"{stem}_transform",
            lambda theta, feats: self._transform_batch(theta, feats, stem, registry),
            [THETA_SQL, struct_in],
            _struct_sql(self.returns),
            type="arrow",
            null_handling="special",
        )


type Foreign = dict[str, Transform]
