"""A transform is a function ``(F, T) -> R`` over relations.

``__FIT__`` and ``__THIS__`` are its two parameters. At the top level ``fit``
binds one and ``transform`` binds the other. Which half is learned and which
is live is read off the text — there is no annotation to remember and none to
forget.

Freezing: every maximal subquery whose leaves are all ``__FIT__`` and
constants is evaluated once at fit and replaced by a table. Those tables are
``params``.

Slice 1 of `docs/superpowers/specs/2026-08-07-datamodel-redesign-design.md`:
single level, no nesting. DuckDB is both the parser and the oracle — a
construct means what DuckDB computes.
"""

from __future__ import annotations

import copy
import json
import sys
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


def _splice(call: Node, scope: dict[str, Any], bindings: dict[str, Any]) -> Node:
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
        relation, arg_depth = _argument(arg, scope, bindings)
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
    _bind_parameters(body, bound)

    ref = _subquery_ref(body, call.get("alias", ""))
    ref["_depth"] = depth + 1
    return ref


def _argument(
    arg: Node, scope: dict[str, Any], bindings: dict[str, Any]
) -> tuple[Node, int]:
    """An argument expression, as the relation it denotes."""
    kind = arg.get("class")
    if kind == "COLUMN_REF" and len(arg["column_names"]) == 1:
        return _base_table(arg["column_names"][0]), 0
    if kind == "SUBQUERY":
        return _subquery_ref(arg["subquery"]["node"], ""), 0
    if kind == "FUNCTION":
        ref = _splice(_table_function_ref(arg, ""), scope, bindings)
        return ref, ref.pop("_depth")
    raise TransformError(
        f"a transform argument is a relation — {FIT}, {THIS}, a parenthesised "
        "query, or another transform call"
    )


def _resolve(doc: Node, scope: dict[str, Any], bindings: dict[str, Any]) -> int:
    """Splice every member call and resolve every free reference, in place.

    Returns the nesting depth. Children are resolved before their parent, so a
    splice at this level always grafts an already-resolved body.
    """
    depth = 0

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
                ref = _splice(v, scope, bindings)
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
    ) -> None:
        self.sql = sql
        self.params = params
        self.bindings = bindings

    def transform(self, data: Any) -> pa.Table:
        con = duckdb.connect()
        con.register(THIS, data)
        for name, table in (self.bindings | self.params).items():
            con.register(name, table)
        return con.execute(self.sql).to_arrow_table()

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
        self.depth = _resolve(doc, scope, self.bindings)
        self.node = doc["statements"][0]["node"]
        self.sql = _deserialize(doc)
        self._steps, self._residual = _plan(copy.deepcopy(doc))

    def fit(self, data: Any) -> Fitted:
        con = duckdb.connect()
        con.register(FIT, data)
        for name, table in self.bindings.items():
            con.register(name, table)
        params: dict[str, pa.Table] = {}
        for param, sql in self._steps:
            # to_arrow_table, never .arrow(): see _duckdb_arrow_test.py — the
            # reader .arrow() returns deadlocks when registered back here.
            params[param] = con.execute(sql).to_arrow_table()
            con.register(param, params[param])
        return Fitted(self._residual, params, self.bindings)

    __call__ = fit


def run(transform: SQLTransform, data: Any) -> pa.Table:
    """Both parameters bound to the same relation, with no freezing at all.

    The reference side of "freezing is faithful". It is a *binding*, not a
    rewrite, which is what keeps that law from restating the implementation.
    """
    con = duckdb.connect()
    con.register(FIT, data)
    con.register(THIS, data)
    for name, table in transform.bindings.items():
        con.register(name, table)
    return con.execute(transform.sql).to_arrow_table()
