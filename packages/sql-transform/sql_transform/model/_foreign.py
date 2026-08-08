"""A foreign transform: the ``(fit, transform)`` pair, supplied directly.

In SQL the pair splits into two DuckDB functions joined by θ, an opaque handle
into a registry of fitted instances. An SQL leaf gives an inspectable params
table; a fitted RandomForest gives a pointer.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import pyarrow as pa

from sql_transform.model._ast import Connection
from sql_transform.model._errors import TransformError

THETA_SQL = "STRUCT(type VARCHAR, id BIGINT)"


THETA_ARROW = pa.struct([("type", pa.string()), ("id", pa.int64())])


def _struct_sql(fields: tuple[str, ...]) -> str:
    for field in fields:
        if not field.isidentifier():
            raise TransformError(f"{field!r} is not a usable struct field name")
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


def _as_matrix(table: pa.Table, takes: tuple[str, ...]) -> Any:
    import numpy as np  # noqa: PLC0415

    return np.column_stack([table[f].to_numpy(zero_copy_only=False) for f in takes])


@dataclass(frozen=True, slots=True)
class _EstimatorFit:
    """``from_estimator``'s fit half, as data rather than a closure.

    Cloned per call, so one estimator object backs many groups without any of
    them sharing learned state — the reason the closure existed.
    """

    estimator: Any
    takes: tuple[str, ...]

    def __call__(self, relation: pa.Table) -> Any:
        from sklearn.base import clone  # noqa: PLC0415

        return clone(self.estimator).fit(_as_matrix(relation, self.takes))


@dataclass(frozen=True, slots=True)
class _EstimatorTransform:
    """``from_estimator``'s transform half. Holds no estimator: the fitted
    instance arrives through θ."""

    takes: tuple[str, ...]
    returns: tuple[str, ...]

    def __call__(self, instance: Any, relation: pa.Table) -> pa.Table:
        import numpy as np  # noqa: PLC0415

        out = np.asarray(instance.transform(_as_matrix(relation, self.takes)))
        return pa.table(
            {name: out[:, i].astype(float) for i, name in enumerate(self.returns)}
        )


@dataclass(slots=True)
class Transform:
    """A foreign transform: the ``(fit, transform)`` pair, supplied directly.

    ``fit(F) -> instance`` and ``transform(instance, T) -> R``, both over
    relations. An sklearn transformer is already this pair — see
    ``from_estimator``.

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

    @classmethod
    def from_estimator(
        cls, estimator: Any, takes: tuple[str, ...], returns: tuple[str, ...]
    ) -> Self:
        """An sklearn transformer as the pair. Cloned per fit, so one
        estimator object can back many groups without sharing learned state.

        The halves are module-level objects holding the estimator as data
        rather than closures over it: a local function is unpicklable, and
        ``deepcopy`` hid that by treating functions as atomic, so ``clone``
        worked while anything that actually serialised — ``Pipeline(memory=)``,
        ``n_jobs>1`` — did not.
        """
        return cls(
            fit=_EstimatorFit(estimator, takes),
            transform=_EstimatorTransform(takes, returns),
            takes=takes,
            returns=returns,
        )

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
