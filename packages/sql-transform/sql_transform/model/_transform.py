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

from __future__ import annotations

import copy
import json
import sys
import threading
from functools import cache
from typing import Any

import duckdb
import pyarrow as pa

FIT = "__FIT__"
THIS = "__THIS__"

MAX_DEPTH = 8

# Only query nodes (SELECT_NODE, SET_OPERATION_NODE, ...) carry a cte_map, so
# its presence is what tells a query node from a table ref or an expression.
_QUERY = "cte_map"

Node = dict[str, Any]


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


def _under(obj: Any, *, deep: bool):
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


def _execute(con: Any, sql: str, registry: _Registry) -> pa.Table:
    """Run ``sql``, letting a foreign transform's own refusal out by name."""
    try:
        # to_arrow_table, never .arrow(): see _duckdb_arrow_test.py — the
        # reader .arrow() returns deadlocks when registered back.
        return con.execute(sql).to_arrow_table()
    except Exception as exc:
        if registry.error is not None:
            raise registry.error from exc
        raise


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

    def __init__(
        self,
        fit: Any,
        transform: Any,
        takes: tuple[str, ...],
        returns: tuple[str, ...],
    ) -> None:
        self.fit = fit
        self.transform = transform
        self.takes = takes
        self.returns = returns
        _struct_sql(takes)
        _struct_sql(returns)

    @staticmethod
    def from_estimator(
        estimator: Any, takes: tuple[str, ...], returns: tuple[str, ...]
    ) -> Transform:
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

        return Transform(fit=fit, transform=transform, takes=takes, returns=returns)

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

    def register(self, con: Any, stem: str, registry: _Registry) -> None:
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


def _splice(
    call: Node,
    scope: dict[str, Any],
    bindings: dict[str, Any],
    foreign: dict[str, Transform],
) -> Node:
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
        relation, arg_depth = _argument(arg, scope, bindings, foreign)
        bound[parameter] = relation
        depth = max(depth, arg_depth)
    if depth + 1 > MAX_DEPTH:
        raise NestingTooDeep(
            f"{name} nests deeper than {MAX_DEPTH} levels of member calls"
        )

    body = copy.deepcopy(member.node)
    renames = {}
    for free, obj in member.bindings.items():
        renames[free] = f"{name}__{free}"
        bindings[f"{name}__{free}"] = obj
    _rename_free(body, renames)

    function_renames = {}
    for stem, leaf in member.foreign.items():
        function_renames[stem] = f"{name}__{stem}"
        foreign[f"{name}__{stem}"] = leaf
    _rename_functions(body, function_renames)

    _bind_parameters(body, bound)

    ref = _subquery_ref(body, call.get("alias", ""))
    ref["_depth"] = depth + 1
    return ref


def _argument(
    arg: Node,
    scope: dict[str, Any],
    bindings: dict[str, Any],
    foreign: dict[str, Transform],
) -> tuple[Node, int]:
    """An argument expression, as the relation it denotes."""
    kind = arg.get("class")
    if kind == "COLUMN_REF" and len(arg["column_names"]) == 1:
        return _base_table(arg["column_names"][0]), 0
    if kind == "SUBQUERY":
        return _subquery_ref(arg["subquery"]["node"], ""), 0
    if kind == "FUNCTION":
        ref = _splice(_table_function_ref(arg, ""), scope, bindings, foreign)
        return ref, ref.pop("_depth")
    raise TransformError(
        f"a transform argument is a relation — {FIT}, {THIS}, a parenthesised "
        "query, or another transform call"
    )


def _resolve(
    doc: Node,
    scope: dict[str, Any],
    bindings: dict[str, Any],
    foreign: dict[str, Transform],
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
        foreign[stem] = member
        if half == "fit":
            # The UDAF half is a scalar function over a collected list.
            call["children"] = [_list_of(child) for child in call["children"]]

    def walk(node: Node, ctes: frozenset[str]) -> None:
        nonlocal depth
        for entry in node[_QUERY]["map"]:
            walk(entry["value"]["query"]["node"], ctes)
            ctes = ctes | {entry["key"]}
        for _, _, v in list(_under(node, deep=False)):
            if _is_query(v):
                walk(v, ctes)
        for parent, key, v in list(_under(node, deep=False)):
            if v.get("type") == "TABLE_FUNCTION":
                if v["function"]["function_name"].lower() in _duckdb_table_functions():
                    continue
                ref = _splice(v, scope, bindings, foreign)
                depth = max(depth, ref.pop("_depth"))
                parent[key] = ref
            elif v.get("type") == "BASE_TABLE":
                name = v["table_name"]
                if name in (FIT, THIS) or name in ctes or name in bindings:
                    continue
                obj = scope.get(name)
                if obj is None:
                    raise UnknownName(
                        f"{name} resolves to nothing in the caller's frame"
                    )
                if isinstance(obj, SQLTransform):
                    raise TransformError(
                        f"{name} is a transform; call it as {name}({FIT}, {THIS})"
                    )
                bindings[name] = obj
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


def _plan(doc: Node) -> tuple[list[tuple[str, str]], str]:
    """Rewrite ``doc`` in place into the residual, returning the fit steps.

    A step is ``(param_name, sql)``, in dependency order: running them against
    a connection with ``__FIT__`` bound, registering each result as it lands,
    produces every table the residual needs. All the analysis — and every
    refusal — happens here, at construction, before any data exists.
    """
    steps: list[tuple[str, str]] = []
    taken: set[str] = set()

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
        if "__param_fit" not in taken:
            taken.add("__param_fit")
            steps.append(("__param_fit", f"SELECT * FROM {FIT}"))  # noqa: S608
        return "__param_fit"

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
            correlated = _correlation(sub, outer)
            if correlated is None:
                parent[key] = freeze(sub, ctes, hint)
                return  # maximal: nothing inside a frozen subtree freezes again
            reference, into_this = correlated
            if into_this:
                raise CorrelatedFit(
                    f"{FIT} subquery references {reference} from the outer "
                    "query, so it cannot be evaluated once into a table"
                )
            # Correlated into a `__FIT__`-only relation instead: still
            # per-outer-row, so still unfreezable, but nothing is wrong — fall
            # through and the training set itself becomes the parameter.
            # `len(params)` reports the cost. Marginalization is what turns it
            # into one row per group.
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
    return steps, _deserialize(doc)


# --- the surface --------------------------------------------------------------


class Fitted:
    """``T -> R``, with the captured environment reified as data.

    A plain closure would be type-correct and unshippable — it could retain
    the whole training set and nothing outside could tell. ``params`` makes
    that a measurement instead of a rule.
    """

    def __init__(
        self,
        sql: str,
        params: dict[str, pa.Table],
        bindings: dict[str, Any],
        foreign: dict[str, Transform],
        instances: dict[int, Any],
    ) -> None:
        self.sql = sql
        self.params = params
        self.bindings = bindings
        self.foreign = foreign
        self.instances = instances

    def transform(self, data: Any) -> pa.Table:
        con = duckdb.connect()
        con.register(THIS, data)
        for name, table in (self.bindings | self.params).items():
            con.register(name, table)
        # A copy: `x_fit` over `__THIS__` is legal and mints instances, but a
        # transductive refit belongs to the call that asked for it.
        registry = _Registry(self.instances)
        for stem, leaf in self.foreign.items():
            leaf.register(con, stem, registry)
        return _execute(con, self.sql, registry)

    __call__ = transform


class SQLTransform:
    """``F -> Fitted``. Construction parses, plans and refuses; nothing else does."""

    def __init__(self, sql: str) -> None:
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

        self.bindings: dict[str, Any] = {}
        self.foreign: dict[str, Transform] = {}
        self.depth = _resolve(doc, scope, self.bindings, self.foreign)
        self.node = doc["statements"][0]["node"]
        self.sql = _deserialize(doc)
        self._steps, self._residual = _plan(copy.deepcopy(doc))

    def _connect(self, registry: _Registry) -> Any:
        con = duckdb.connect()
        for name, table in self.bindings.items():
            con.register(name, table)
        for stem, leaf in self.foreign.items():
            leaf.register(con, stem, registry)
        return con

    def fit(self, data: Any) -> Fitted:
        registry = _Registry()
        con = self._connect(registry)
        con.register(FIT, data)
        params: dict[str, pa.Table] = {}
        for param, sql in self._steps:
            params[param] = _execute(con, sql, registry)
            con.register(param, params[param])
        return Fitted(
            self._residual, params, self.bindings, self.foreign, registry.instances
        )

    __call__ = fit


def run(transform: SQLTransform, data: Any) -> pa.Table:
    """Both parameters bound to the same relation, with no freezing at all.

    The reference side of "freezing is faithful". It is a *binding*, not a
    rewrite, which is what keeps that law from restating the implementation.
    """
    registry = _Registry()
    con = transform._connect(registry)
    con.register(FIT, data)
    con.register(THIS, data)
    return _execute(con, transform.sql, registry)
