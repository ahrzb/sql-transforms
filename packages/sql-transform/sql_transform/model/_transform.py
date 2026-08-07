"""A transform is a function ``(F, T) -> R`` over relations.

``__FIT__`` and ``__THIS__`` are its two parameters. At the top level ``fit``
binds one and ``transform`` binds the other. Which half is learned and which
is live is read off the text — there is no annotation to remember and none to
forget.

Freezing: every maximal subquery whose leaves are all ``__FIT__`` and
constants is evaluated once at fit and replaced by a table. Those tables are
``params``.

Implements `docs/superpowers/specs/2026-08-07-datamodel-redesign-design.md`:
the two parameters and freezing, calling (nesting, chaining, per group), and
foreign transforms. DuckDB is both the parser and the oracle — a construct
means what DuckDB computes.
"""

import copy
import itertools
import json
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from typing import Any, Self

import duckdb
import pyarrow as pa

FIT = "__FIT__"
THIS = "__THIS__"

MAX_DEPTH = 8

# Only query nodes (SELECT_NODE, SET_OPERATION_NODE, ...) carry a cte_map, so
# its presence is what tells a query node from a table ref or an expression.
# A TableRef is then anything carrying a `sample`. Both are internal DuckDB
# details rather than a documented API; `_shapes_test.py` is what keeps them
# true, and replays DuckDB's own corpus to catch the format moving.
_QUERY = "cte_map"
_RECURSIVE_CTE = "RECURSIVE_CTE_NODE"

type Node = dict[str, Any]
type Relation = Any  # anything DuckDB will register: arrow, pandas, polars
type Connection = duckdb.DuckDBPyConnection
type LazyRelation = duckdb.DuckDBPyRelation
type Params = dict[str, pa.Table]
type Bindings = dict[str, Relation]
type Foreign = dict[str, Transform]
# Everything a transform resolved from the caller's frame, by name: lookup
# relations, foreign transforms, and members. One map so `clone` can carry
# the lot; the two views above are derived from it.
type Captured = dict[str, Any]


class TransformError(Exception):
    """Every refusal. Named, and raised at construction (P7)."""


class CorrelatedFit(TransformError):
    """A ``__FIT__`` subtree correlates out of itself.

    It is per-outer-row, so it cannot be evaluated once into a table.
    Supporting it means lifting the correlation to a ``GROUP BY`` and
    rewriting it as a join — which is marginalization. Future work, not a
    permanent boundary.
    """


class UnknownName(TransformError):
    """An identifier resolved to nothing in the caller's frame."""


class NestingTooDeep(TransformError):
    """More than ``MAX_DEPTH`` levels of member calls."""


class NotFitted(TransformError):
    """The estimator surface was used before ``fit``.

    Ours rather than sklearn's ``NotFittedError``, because sklearn is not a
    runtime dependency — a SQL transform must not need one to refuse.
    """


# --- the oracle as parser and printer ----------------------------------------


def _serialize(sql: str) -> Node:
    (out,) = duckdb.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()
    doc = json.loads(out)
    if doc.get("error"):
        raise TransformError(
            f"parse error at position {doc.get('position')}: {doc.get('error_message')}"
        )
    return doc


def _deserialize(doc: Node) -> str:
    (out,) = duckdb.execute(
        "SELECT json_deserialize_sql(?::JSON)", [json.dumps(doc)]
    ).fetchone()
    return out


def _statement(node: Node) -> Node:
    return {"error": False, "statements": [{"node": node, "named_param_map": []}]}


def _select_star(name: str) -> Node:
    """A query node reading nothing but ``name``. Cut from the oracle's own
    serialization so it carries every field the deserializer expects."""
    node = _template("SELECT * FROM __tpl__")
    node["from_table"]["table_name"] = name
    return node


@cache
def _cached_template(sql: str) -> str:
    return json.dumps(_serialize(sql)["statements"][0]["node"])


def _template(sql: str) -> Node:
    """A fresh copy of a node shape, cut from the oracle's own serialization
    so a grafted node always carries every field the deserializer expects."""
    return json.loads(_cached_template(sql))


def _base_table(name: str, alias: str = "") -> Node:
    ref = _template("SELECT * FROM __tpl__")["from_table"]
    ref["table_name"] = name
    ref["alias"] = alias
    return ref


def _subquery_ref(node: Node, alias: str) -> Node:
    ref = _template("SELECT * FROM (SELECT 1) __tpl__")["from_table"]
    ref["subquery"]["node"] = node
    ref["alias"] = alias
    return ref


def _table_function_ref(function: Node, alias: str) -> Node:
    ref = _template("SELECT * FROM range(1)")["from_table"]
    ref["function"] = function
    ref["alias"] = alias
    return ref


@cache
def _duckdb_table_functions() -> frozenset[str]:
    """Names the oracle already owns. A table call outside this set is a
    member call, resolved against the caller's frame, never guessed."""
    rows = duckdb.execute(
        "SELECT DISTINCT function_name FROM duckdb_functions()"
        " WHERE function_type = 'table'"
    ).fetchall()
    return frozenset(name.lower() for (name,) in rows)


def normalize(sql: str) -> str:
    """``sql`` as the oracle prints it. The only way to compare SQL text."""
    return _deserialize(_serialize(sql))


# --- walking ------------------------------------------------------------------


def _is_query(value: Any) -> bool:
    return isinstance(value, dict) and _QUERY in value


def _under(obj: Any, *, deep: bool) -> Iterator[tuple[Any, Any, Node]]:
    """Yield ``(parent, key, dict)`` for every dict below ``obj``.

    ``deep=False`` stops at nested query nodes — it still yields them, so a
    caller can rewrite the slot, but does not look inside. It also skips
    ``cte_map``, which callers walk in definition order instead.
    """
    if isinstance(obj, dict):
        items: Any = list(obj.items())
    elif isinstance(obj, list):
        items = list(enumerate(obj))
    else:
        return
    for key, value in items:
        if key == _QUERY and not deep:
            continue
        if isinstance(value, dict):
            yield obj, key, value
            if _is_query(value) and not deep:
                continue
        if isinstance(value, dict | list):
            yield from _under(value, deep=deep)


def _is_ref(value: Node) -> bool:
    """A TableRef. Only those carry a ``sample``; query nodes do too, so the
    discriminator is `has sample and is not a query node`."""
    return "sample" in value and not _is_query(value)


def _reads(node: Node) -> set[str]:
    """Which of the two parameters the whole subtree reads. ``node`` itself
    counts — a bare ``FROM __THIS__`` ref is the whole subtree."""
    seen = set()
    for v in (node, *(v for _, _, v in _under(node, deep=True))):
        if v.get("type") == "BASE_TABLE" and v.get("table_name") in (FIT, THIS):
            seen.add(v["table_name"])
    return seen


def _names_in(node: Node) -> set[str]:
    """Every name a column reference could be qualified by, anywhere inside."""
    names = {e["key"] for e in node[_QUERY]["map"]}
    for _, _, v in _under(node, deep=True):
        if _is_ref(v):
            names.add(v.get("alias") or v.get("table_name") or "")
        if _is_query(v):
            names.update(e["key"] for e in v[_QUERY]["map"])
    names.discard("")
    return names


def _bindings_at(node: Node) -> dict[str, bool]:
    """The names this level binds, each mapped to *does it read ``__THIS__``*.

    Which side a correlation lands on is what separates the one refusal from
    the case that merely costs params.
    """
    out = {
        e["key"]: THIS in _reads(e["value"]["query"]["node"])
        for e in node[_QUERY]["map"]
    }
    for _, _, v in _under(node, deep=False):
        if _is_ref(v):
            alias = v.get("alias") or v.get("table_name") or ""
            if alias:
                out[alias] = THIS in _reads(v)
    return out


def _correlation(node: Node, outer: dict[str, bool]) -> tuple[str, bool] | None:
    """A column reference reaching out of ``node``, and whether its target
    reads ``__THIS__``.

    Only qualified references are checked. An unqualified one is ambiguous
    without a binder, and DuckDB resolves it inward whenever it can.
    """
    inside = _names_in(node)
    found: tuple[str, bool] | None = None
    for _, _, v in _under(node, deep=True):
        if v.get("class") != "COLUMN_REF":
            continue
        parts = v["column_names"]
        if len(parts) >= 2 and parts[0] not in inside and parts[0] in outer:
            if outer[parts[0]]:
                return ".".join(parts), True  # into __THIS__: the one refusal
            found = found or (".".join(parts), False)
    return found


# --- foreign transforms -------------------------------------------------------

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
        estimator object can back many groups without sharing learned state."""
        import numpy as np  # noqa: PLC0415
        from sklearn.base import clone  # noqa: PLC0415

        def as_matrix(table: pa.Table) -> Any:
            return np.column_stack(
                [table[f].to_numpy(zero_copy_only=False) for f in takes]
            )

        def fit(relation: pa.Table) -> Any:
            return clone(estimator).fit(as_matrix(relation))

        def transform(instance: Any, relation: pa.Table) -> pa.Table:
            out = np.asarray(instance.transform(as_matrix(relation)))
            return pa.table(
                {name: out[:, i].astype(float) for i, name in enumerate(returns)}
            )

        return cls(fit=fit, transform=transform, takes=takes, returns=returns)

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
            thetas.append({"type": stem, "id": registry.add(self.fit(relation))})
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
            produced = self.transform(registry.instances[iid], relation)
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


@cache
def _duckdb_functions() -> frozenset[str]:
    """Every function the oracle knows, of any type. A call outside this set
    is a foreign candidate — resolved against the caller's frame, never
    guessed."""
    rows = duckdb.execute(
        "SELECT DISTINCT function_name FROM duckdb_functions()"
    ).fetchall()
    return frozenset(name.lower() for (name,) in rows)


def _list_of(argument: Node) -> Node:
    """``list(arg)``. DuckDB's Python API has no aggregate UDF, so the UDAF
    half is a scalar function over the list DuckDB's own ``list()`` collects.
    The author's text is unchanged; ``GROUP BY`` still does the grouping."""
    call = _template("SELECT list(1)")["select_list"][0]
    call["children"] = [argument]
    call["alias"] = ""
    return call


# --- calling ------------------------------------------------------------------


def _rename_free(body: Node, renames: dict[str, str]) -> None:
    """Rename the member's *free* references, not its own definitions.

    A parenthesised derived table already scopes its own CTEs, so the member's
    definitions cannot collide with the caller's. The hazard runs the other
    way: a free name the member resolved from its own frame gets captured by
    an outer CTE that happens to share it. Measured — silent, no error,
    different numbers. The old name survives as the alias, so every column
    reference qualified by it still resolves.
    """
    for _, _, v in list(_under(body, deep=True)):
        if v.get("type") == "BASE_TABLE" and v.get("table_name") in renames:
            old = v["table_name"]
            v["table_name"] = renames[old]
            v["alias"] = v.get("alias") or old


def _bind_parameters(body: Node, args: dict[str, Node]) -> None:
    """Bind the member's two parameters to the call site's arguments.

    Sites are collected before any replacement, so an argument that itself
    mentions ``__FIT__`` — every chained call does — is never re-substituted.
    ``_under`` happens to be safe against that anyway, since it lists each
    dict's items before descending; collecting first is what keeps the
    guarantee from resting on that detail.
    """
    sites = [
        (parent, key, v)
        for parent, key, v in _under(body, deep=True)
        if v.get("type") == "BASE_TABLE" and v.get("table_name") in args
    ]
    for parent, key, ref in sites:
        bound = copy.deepcopy(args[ref["table_name"]])
        bound["alias"] = ref.get("alias") or ref["table_name"]
        parent[key] = bound


def _rename_functions(body: Node, renames: dict[str, str]) -> None:
    """Rename the member's foreign calls the same way as its free tables, so
    two members can each carry their own ``sc`` without colliding."""
    for _, _, v in list(_under(body, deep=True)):
        if v.get("class") != "FUNCTION":
            continue
        stem, _, half = v["function_name"].rpartition("_")
        if half in ("fit", "transform") and stem in renames:
            v["function_name"] = f"{renames[stem]}_{half}"


def _splice(call: Node, scope: dict[str, Any], captured: Captured) -> Node:
    """A member call, as the spliced relation it denotes.

    Splice, never emit a DuckDB macro: measured, a table macro invoked under
    ``LATERAL`` does not see the correlation and silently returns the
    whole-table answer for every group.
    """
    function = call["function"]
    name = function["function_name"]
    member = scope.get(name)
    if member is None:
        raise UnknownName(
            f"{name} is not a table function and resolves to nothing in the "
            "caller's frame"
        )
    if not isinstance(member, SQLTransform):
        raise TransformError(
            f"{name} resolves to a {type(member).__name__}, not a transform"
        )
    args = function["children"]
    if len(args) != 2:
        raise TransformError(
            f"a transform takes two arguments ({FIT}, {THIS}); "
            f"{name} was called with {len(args)}"
        )

    depth = member.depth
    bound = {}
    for parameter, arg in zip((FIT, THIS), args, strict=True):
        relation, arg_depth = _argument(arg, scope, captured)
        bound[parameter] = relation
        depth = max(depth, arg_depth)
    if depth + 1 > MAX_DEPTH:
        raise NestingTooDeep(
            f"{name} nests deeper than {MAX_DEPTH} levels of member calls"
        )

    captured[name] = member  # spliced away, but a clone has to find it again

    body = copy.deepcopy(member.node)
    renames = {}
    for free, obj in member.bindings.items():
        renames[free] = f"{name}__{free}"
        captured[f"{name}__{free}"] = obj
    _rename_free(body, renames)

    function_renames = {}
    for stem, leaf in member.foreign.items():
        function_renames[stem] = f"{name}__{stem}"
        captured[f"{name}__{stem}"] = leaf
    _rename_functions(body, function_renames)

    _bind_parameters(body, bound)

    ref = _subquery_ref(body, call.get("alias", ""))
    ref["_depth"] = depth + 1
    return ref


def _argument(arg: Node, scope: dict[str, Any], captured: Captured) -> tuple[Node, int]:
    """An argument expression, as the relation it denotes."""
    match arg:
        case {"class": "COLUMN_REF", "column_names": [name]}:
            return _base_table(name), 0
        case {"class": "SUBQUERY", "subquery": {"node": node}}:
            return _subquery_ref(node, ""), 0
        case {"class": "FUNCTION"}:
            ref = _splice(_table_function_ref(arg, ""), scope, captured)
            return ref, ref.pop("_depth")
    raise TransformError(
        f"a transform argument is a relation — {FIT}, {THIS}, a parenthesised "
        "query, or another transform call"
    )


def _catalog(con: Connection | None) -> frozenset[str]:
    """Tables and views the supplied connection already owns.

    Passing a connection is how you say *use my catalog*, so a name it can
    already resolve is not a free reference and must not be looked for in
    the caller's frame — nor renamed when a shared connection is mangled.
    """
    if con is None:
        return frozenset()
    rows = con.execute(
        "SELECT table_name FROM duckdb_tables()"
        " UNION ALL SELECT view_name FROM duckdb_views()"
    ).fetchall()
    return frozenset(name for (name,) in rows)


def _resolve(
    doc: Node,
    scope: dict[str, Any],
    captured: Captured,
    catalog: frozenset[str] = frozenset(),
) -> int:
    """Splice every member call and resolve every free name, in place.

    Returns the nesting depth. Children are resolved before their parent, so a
    splice at this level always grafts an already-resolved body.
    """
    depth = 0

    def foreign_call(call: Node) -> None:
        """``x_fit``/``x_transform``: the stem resolves, the suffix says half."""
        name = call["function_name"]
        stem, _, half = name.rpartition("_")
        member = scope.get(stem) if half in ("fit", "transform") else None
        if not isinstance(member, Transform):
            raise UnknownName(
                f"{name} is not a DuckDB function, and "
                + (
                    f"{stem} resolves to nothing in the caller's frame"
                    if member is None
                    else f"{stem} resolves to a {type(member).__name__}, "
                    "not a Transform"
                )
            )
        captured[stem] = member
        if half == "fit":
            # The UDAF half is a scalar function over a collected list.
            call["children"] = [_list_of(child) for child in call["children"]]

    def walk(node: Node, ctes: frozenset[str]) -> None:
        nonlocal depth
        for entry in node[_QUERY]["map"]:
            # DuckDB would let such a CTE win, and we would go on rewriting the
            # reference to the training set — two meanings for one name, and
            # the row count changed with no error. Refused where it is
            # defined, so `__FIT__` means the parameter everywhere or the text
            # does not compile.
            if entry["key"].upper() in (FIT, THIS):
                raise TransformError(
                    f"a CTE may not be named {entry['key']!r}: {FIT} and "
                    f"{THIS} are the transform's two parameters"
                )
            body = entry["value"]["query"]["node"]
            # A RECURSIVE CTE is in scope inside its own body; a plain one is
            # not, where the same name means whatever the caller's frame binds.
            # The inner node type is the only thing that tells them apart.
            visible = ctes
            if body["type"] == _RECURSIVE_CTE:
                visible = ctes | {entry["key"]}
            walk(body, visible)
            ctes = ctes | {entry["key"]}
        for _, _, v in list(_under(node, deep=False)):
            if _is_query(v):
                walk(v, ctes)
        for parent, key, v in list(_under(node, deep=False)):
            match v:
                case {"type": "TABLE_FUNCTION", "function": {"function_name": call}}:
                    if call.lower() in _duckdb_table_functions():
                        continue
                    ref = _splice(v, scope, captured)
                    depth = max(depth, ref.pop("_depth"))
                    parent[key] = ref
                case {"type": "BASE_TABLE", "table_name": name} if (
                    name not in (FIT, THIS)
                    and name not in ctes
                    and name not in captured
                    and name not in catalog
                ):
                    match scope.get(name):
                        case None:
                            raise UnknownName(
                                f"{name} resolves to nothing in the caller's frame"
                            )
                        case SQLTransform():
                            raise TransformError(
                                f"{name} is a transform; call it as "
                                f"{name}({FIT}, {THIS})"
                            )
                        case obj:
                            captured[name] = obj
        # Member calls are gone by now, so every FUNCTION left at this level is
        # a scalar one — no need to tell the table call's own function apart.
        for _, _, v in list(_under(node, deep=False)):
            if (
                v.get("class") == "FUNCTION"
                and not v.get("is_operator")
                and v["function_name"].lower() not in _duckdb_functions()
            ):
                foreign_call(v)

    walk(doc["statements"][0]["node"], frozenset())
    return depth


# --- the plan -----------------------------------------------------------------


def _plan(doc: Node) -> tuple[list[tuple[str, str]], Node]:
    """Rewrite ``doc`` in place into the residual, returning the fit steps.

    A step is ``(param_name, sql)``, in dependency order: running them against
    a connection with ``__FIT__`` bound, registering each result as it lands,
    produces every table the residual needs. All the analysis — and every
    refusal — happens here, at construction, before any data exists.
    """
    steps: list[tuple[str, str]] = []
    taken: set[str] = set()
    whole: str | None = None

    def name(hint: str | None) -> str:
        base = f"__param_{hint}" if hint else f"__param_{len(steps)}"
        candidate, n = base, 1
        while candidate in taken:
            candidate, n = f"{base}_{n}", n + 1
        taken.add(candidate)
        return candidate

    def whole_fit() -> str:
        # A bare `FROM __FIT__` inside a relation that also reads `__THIS__`.
        # The training set itself is the parameter; `len(params)` says so.
        #
        # `whole` rather than a membership test on `taken`: hints come from CTE
        # keys, so a CTE named `fit` mints `__param_fit` too. Asking whether
        # the *name* was taken conflated "someone else has it" with "my step is
        # already emitted", and the bare `FROM __FIT__` was then aliased onto
        # that CTE's table — silently, and only when a CTE happened to be
        # called `fit`. Emitted-ness is its own fact; the name comes from the
        # same collision-avoiding allocator as every other one.
        nonlocal whole
        if whole is None:
            whole = name("fit")
            steps.append((whole, f"SELECT * FROM {FIT}"))  # noqa: S608
        return whole

    def freeze(sub: Node, ctes: list[Node], hint: str | None) -> Node:
        frozen = copy.deepcopy(sub)
        # Carry the enclosing CTEs in: a frozen subtree may reference one, and
        # by now their own definitions have been rewritten to frozen tables.
        frozen[_QUERY]["map"] = copy.deepcopy(ctes) + frozen[_QUERY]["map"]
        param = name(hint)
        steps.append((param, _deserialize(_statement(frozen))))
        return _select_star(param)

    def visit(
        parent: Node,
        key: str,
        ctes: list[Node],
        outer: dict[str, bool],
        hint: str | None,
    ) -> None:
        sub = parent[key]
        reads = _reads(sub)
        if FIT in reads and THIS not in reads:
            match _correlation(sub, outer):
                case None:
                    parent[key] = freeze(sub, ctes, hint)
                    return  # maximal: a frozen subtree never refreezes
                case (reference, True):
                    raise CorrelatedFit(
                        f"{FIT} subquery references {reference} from the "
                        "outer query, so it cannot be evaluated once into "
                        "a table"
                    )
                case _:
                    # Correlated into a `__FIT__`-only relation instead:
                    # still per-outer-row, so still unfreezable, but
                    # nothing is wrong. Fall through and the training set
                    # itself becomes the parameter; `len(params)` reports
                    # the cost. Marginalization makes it one row per group.
                    pass
        descend(sub, ctes, outer)

    def descend(node: Node, ctes: list[Node], outer: dict[str, bool]) -> None:
        ctes = list(ctes)
        for entry in node[_QUERY]["map"]:
            visit(entry["value"]["query"], "node", ctes, outer, entry["key"])
            ctes.append(entry)
        outer = outer | _bindings_at(node)
        sites = [(p, k) for p, k, v in _under(node, deep=False) if _is_query(v)]
        for parent, key in sites:
            visit(parent, key, ctes, outer, None)
        for _, _, v in _under(node, deep=False):
            if v.get("type") == "BASE_TABLE" and v.get("table_name") == FIT:
                v["table_name"] = whole_fit()

    visit(doc["statements"][0], "node", [], {}, None)
    return steps, doc["statements"][0]["node"]


# --- the surface --------------------------------------------------------------


@dataclass(slots=True, eq=False, repr=False)
class Fitted:
    """``T -> R``, with the captured environment reified as data.

    A plain closure would be type-correct and unshippable — it could retain
    the whole training set and nothing outside could tell. ``params`` makes
    that a measurement instead of a rule.
    """

    node: Node
    params: Params
    bindings: Bindings
    foreign: Foreign
    instances: dict[int, Any]
    connection: Connection | None = None

    def __repr__(self) -> str:
        # The generated one prints the whole residual AST: unreadable, and
        # it buries the two numbers that actually say what you are holding.
        shape = ", ".join(f"{k}[{len(v)}]" for k, v in self.params.items())
        return f"Fitted(params={shape or 'none'}, instances={len(self.instances)})"

    @property
    def sql(self) -> str:
        """The residual, under the names ``params`` uses. What you read."""
        return _deserialize(_statement(self.node))

    def _bind(self, data: Relation) -> tuple[Connection, str, _Registry]:
        """Register everything this residual needs, and say what to execute.

        On a connection we own, names are the readable ones and a fresh
        connection per call makes collisions impossible.

        On a *shared* connection they cannot be: two transforms in a pipeline
        both bind `__THIS__` and both call a parameter `__param_0`. Eagerly
        that is harmless — each materialises before the next registers — but a
        lazy relation is not executed yet, so stage one would end up reading
        stage two's tables. Same shape, different numbers, no error. So a
        shared connection gets one rename per execution.
        """
        registry = _Registry(self.instances)
        tables = {THIS: data} | self.bindings | self.params
        if self.connection is None:
            con = duckdb.connect()
            for name, table in tables.items():
                con.register(name, table)
            for stem, leaf in self.foreign.items():
                leaf.register(con, stem, registry)
            return con, self.sql, registry

        con = self.connection
        token = next(_EXECUTIONS)
        renames = {name: f"{name}__x{token}" for name in tables}
        functions = {stem: f"{stem}__x{token}" for stem in self.foreign}
        doc = copy.deepcopy(_statement(self.node))
        _rename_free(doc, renames)
        _rename_functions(doc, functions)
        for name, table in tables.items():
            con.register(renames[name], table)
        for stem, leaf in self.foreign.items():
            leaf.register(con, functions[stem], registry)
        return con, _deserialize(doc), registry

    def transform(self, data: Relation) -> pa.Table:
        con, sql, registry = self._bind(data)
        return _execute(con, sql, registry)

    __call__ = transform

    def relation(self, data: Relation) -> LazyRelation:
        """The residual as an unexecuted ``DuckDBPyRelation``.

        Nothing is materialised: bind, plan, hand it back. DuckDB still
        *binds* eagerly, so an unknown column refuses here; a foreign
        transform's refusal only surfaces when the relation is consumed, which
        is the price of not materialising.

        A relation belongs to the connection that built it and cannot be
        handed to another one, not even to a cursor of the same connection.
        Chaining lazily therefore means giving both transforms the same
        ``connection=``.
        """
        con, sql, _ = self._bind(data)
        return con.sql(sql)


OUTPUTS = ("default", "arrow", "duckdb", "pandas", "numpy")

# One counter per process, so a shared connection never sees two executions
# under the same name. See Fitted._bind.
_EXECUTIONS = itertools.count()


def _as_output(table: pa.Table, output: str) -> Any:
    match output:
        case "default" | "arrow":
            return table
        case "pandas":
            return table.to_pandas()
        case "numpy":
            return table.to_pandas().to_numpy()
    raise TransformError(f"output must be one of {OUTPUTS}; got {output!r}")


class SQLTransform:
    """``F -> Fitted``, and an sklearn estimator.

    ``fit`` returns the ``Fitted`` artifact rather than ``self``. That is the
    currying the model is built on — ``.params`` is a thing you can ship —
    and it costs nothing with sklearn, which never reads what ``fit``
    returned: ``Pipeline`` keeps the object it called and asks *it* to
    ``transform`` later. So ``fit`` also remembers, and both spellings agree:

        t.fit(D).transform(X)     # curried: the artifact transforms
        t.fit(D); t.transform(X)  # stateful: the estimator transforms

    ``bindings`` and ``foreign`` are constructor parameters as well as frame
    lookups, because ``clone`` rebuilds an estimator inside sklearn's own
    frame, where a member or a lookup table is not in scope. They ride along
    in ``get_params`` so a clone resolves to the very same objects.

    Those two mappings are *adopted*, not copied, and completed in place with
    whatever the frame supplied. ``clone`` demands that ``get_params`` hand
    back the very object the constructor was given — a defensive copy fails
    its identity check — and carrying the completed set is the whole point.

    Construction parses, plans and refuses; nothing else does.
    """

    def __init__(
        self,
        sql: str,
        output: str = "default",
        connection: Connection | None = None,
        captured: Captured | None = None,
    ) -> None:
        doc = _serialize(sql)
        if len(doc["statements"]) != 1:
            raise TransformError(
                f"a transform is one statement, got {len(doc['statements'])}"
            )
        # Resolution happens once, here, and captures by value: `scope` is a
        # local that dies with this call, so no caller frame is retained and
        # rebinding a member afterwards cannot change what was built.
        frame = sys._getframe(1)
        scope = frame.f_globals | frame.f_locals
        del frame

        self.output = output
        # Given rather than conjured. A transform that makes its own hidden
        # connection cannot compose with anything: a DuckDBPyRelation belongs
        # to the connection that built it, so lazy output only chains when
        # both stages share one. Pass it and you own it.
        self.connection = connection
        # Adopted, not copied: see the class docstring. Explicit entries win,
        # and are how a clone keeps names the frame it was rebuilt in cannot
        # see.
        self.captured: Captured = {} if captured is None else captured
        scope = scope | self.captured

        self.depth = _resolve(doc, scope, self.captured, _catalog(connection))
        # Two runtime views. A member is spliced away, so it is neither.
        self.foreign: Foreign = {
            k: v for k, v in self.captured.items() if isinstance(v, Transform)
        }
        self.bindings: Bindings = {
            k: v
            for k, v in self.captured.items()
            if not isinstance(v, Transform | SQLTransform)
        }
        self.node = doc["statements"][0]["node"]
        self.source = sql  # the exact object, so clone's identity check passes
        self.sql = _deserialize(doc)
        self._steps, self._residual = _plan(copy.deepcopy(doc))
        self._own = connection is None
        self.fitted_: Fitted | None = None
        self.feature_names_out_: list[str] | None = None

    # -- the model's own surface ----------------------------------------------

    def __repr__(self) -> str:
        state = "fitted" if self.fitted_ is not None else "unfitted"
        return f"SQLTransform({self.sql!r}, output={self.output!r}, {state})"

    def _connect(self, registry: _Registry) -> Connection:
        con = duckdb.connect() if self._own else self.connection
        for name, table in self.bindings.items():
            con.register(name, table)
        for stem, leaf in self.foreign.items():
            leaf.register(con, stem, registry)
        return con

    def fit(self, data: Relation, y: Any = None) -> Fitted:
        """Partial application — and the estimator remembers the result.

        ``y`` is accepted and ignored: a target belongs in the relation, as a
        column ``__FIT__`` can read, not in a second argument the SQL cannot
        name.
        """
        registry = _Registry()
        con = self._connect(registry)
        con.register(FIT, data)
        params: Params = {}
        for param, sql in self._steps:
            params[param] = _execute(con, sql, registry)
            con.register(param, params[param])
        self.fitted_ = Fitted(
            self._residual,
            params,
            self.bindings,
            self.foreign,
            registry.instances,
            self.connection,
        )
        return self.fitted_

    __call__ = fit

    @property
    def params_(self) -> Params:
        return self._require_fit().params

    @property
    def instances_(self) -> dict[int, Any]:
        return self._require_fit().instances

    # -- the sklearn surface ---------------------------------------------------

    def _require_fit(self) -> Fitted:
        if self.fitted_ is None:
            raise NotFitted("this transform has not been fit; call fit first")
        return self.fitted_

    def transform(self, data: Relation) -> Any:
        fitted = self._require_fit()
        if self.output == "duckdb":
            lazy = fitted.relation(data)  # the whole point: never materialise
            self.feature_names_out_ = list(lazy.columns)
            return lazy
        out = fitted.transform(data)
        self.feature_names_out_ = out.column_names
        return _as_output(out, self.output)

    def fit_transform(self, data: Relation, y: Any = None) -> Any:
        """On the training relation this is exactly ``run(t, D)`` — that is
        the *freezing is faithful* law, not a coincidence."""
        self.fit(data)
        return self.transform(data)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        if self.feature_names_out_ is None:
            raise NotFitted(
                "output column names are only known once something has been "
                "transformed; call transform or fit_transform first"
            )
        return list(self.feature_names_out_)

    def set_output(self, *, transform: str | None = None) -> Self:
        """sklearn's opt-in: ``pandas`` or ``numpy`` for a downstream
        estimator, ``default`` for the model's own arrow tables."""
        if transform is not None:
            if transform not in OUTPUTS:
                raise TransformError(
                    f"output must be one of {OUTPUTS}; got {transform!r}"
                )
            self.output = transform
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "sql": self.source,
            "output": self.output,
            "connection": self.connection,
            "captured": self.captured,
        }

    def set_params(self, **params: Any) -> Self:
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise TransformError(f"unknown parameters {sorted(unknown)}")
        if {"sql", "captured", "connection"} & set(params):
            # The plan is derived from all three, so rebuild rather than let
            # them drift apart.
            rebuilt = type(self)(
                params.get("sql", self.source),
                output=params.get("output", self.output),
                connection=params.get("connection", self.connection),
                captured=params.get("captured", self.captured),
            )
            self.__dict__.update(rebuilt.__dict__)
        elif "output" in params:
            self.set_output(transform=params["output"])
        return self


def run(transform: SQLTransform, data: Relation) -> pa.Table:
    """Both parameters bound to the same relation, with no freezing at all.

    The reference side of "freezing is faithful". It is a *binding*, not a
    rewrite, which is what keeps that law from restating the implementation.
    """
    registry = _Registry()
    con = transform._connect(registry)
    con.register(FIT, data)
    con.register(THIS, data)
    return _execute(con, transform.sql, registry)
