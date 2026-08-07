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
from typing import Any

import duckdb
import pyarrow as pa

FIT = "__FIT__"
THIS = "__THIS__"

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
    # S608: `name` is a param name this module minted, never user input.
    return _serialize(f"SELECT * FROM {name}")["statements"][0]["node"]  # noqa: S608


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


def _reads(node: Node) -> set[str]:
    """Which of the two parameters the whole subtree reads."""
    return {
        v["table_name"]
        for _, _, v in _under(node, deep=True)
        if v.get("type") == "BASE_TABLE" and v.get("table_name") in (FIT, THIS)
    }


def _bound(node: Node, *, deep: bool) -> set[str]:
    """Every name a column reference could be qualified by at this level."""
    names = {e["key"] for e in node[_QUERY]["map"]}
    for _, _, v in _under(node, deep=deep):
        if "sample" in v and not _is_query(v):  # a TableRef
            names.add(v.get("alias") or v.get("table_name") or "")
        if deep and _is_query(v):
            names.update(e["key"] for e in v[_QUERY]["map"])
    names.discard("")
    return names


def _correlation(node: Node, outer: set[str]) -> str | None:
    """The first column reference reaching out of ``node`` into ``outer``.

    Only qualified references are checked. An unqualified one is ambiguous
    without a binder, and DuckDB resolves it inward whenever it can.
    """
    inside = _bound(node, deep=True)
    for _, _, v in _under(node, deep=True):
        if v.get("class") != "COLUMN_REF":
            continue
        parts = v["column_names"]
        if len(parts) >= 2 and parts[0] not in inside and parts[0] in outer:
            return ".".join(parts)
    return None


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
        parent: Node, key: str, ctes: list[Node], outer: set[str], hint: str | None
    ) -> None:
        sub = parent[key]
        reads = _reads(sub)
        if FIT in reads and THIS not in reads:
            correlated = _correlation(sub, outer)
            if correlated is not None:
                raise CorrelatedFit(
                    f"__FIT__ subquery references {correlated} from the outer "
                    "query, so it cannot be evaluated once into a table"
                )
            parent[key] = freeze(sub, ctes, hint)
            return  # maximal: nothing inside a frozen subtree is frozen again
        descend(sub, ctes, outer)

    def descend(node: Node, ctes: list[Node], outer: set[str]) -> None:
        ctes = list(ctes)
        for entry in node[_QUERY]["map"]:
            visit(entry["value"]["query"], "node", ctes, outer, entry["key"])
            ctes.append(entry)
        outer = outer | _bound(node, deep=False)
        sites = [(p, k) for p, k, v in _under(node, deep=False) if _is_query(v)]
        for parent, key in sites:
            visit(parent, key, ctes, outer, None)
        for _, _, v in _under(node, deep=False):
            if v.get("type") == "BASE_TABLE" and v.get("table_name") == FIT:
                v["table_name"] = whole_fit()

    visit(doc["statements"][0], "node", [], set(), None)
    return steps, _deserialize(doc)


# --- the surface --------------------------------------------------------------


class Fitted:
    """``T -> R``, with the captured environment reified as data.

    A plain closure would be type-correct and unshippable — it could retain
    the whole training set and nothing outside could tell. ``params`` makes
    that a measurement instead of a rule.
    """

    def __init__(self, sql: str, params: dict[str, pa.Table]) -> None:
        self.sql = sql
        self.params = params

    def transform(self, data: Any) -> pa.Table:
        con = duckdb.connect()
        con.register(THIS, data)
        for param, table in self.params.items():
            con.register(param, table)
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
        self.sql = _deserialize(doc)
        self._steps, self._residual = _plan(copy.deepcopy(doc))

    def fit(self, data: Any) -> Fitted:
        con = duckdb.connect()
        con.register(FIT, data)
        params: dict[str, pa.Table] = {}
        for param, sql in self._steps:
            # to_arrow_table, never .arrow(): see _duckdb_arrow_test.py — the
            # reader .arrow() returns deadlocks when registered back here.
            params[param] = con.execute(sql).to_arrow_table()
            con.register(param, params[param])
        return Fitted(self._residual, params)

    __call__ = fit


def run(transform: SQLTransform, data: Any) -> pa.Table:
    """Both parameters bound to the same relation, with no freezing at all.

    The reference side of "freezing is faithful". It is a *binding*, not a
    rewrite, which is what keeps that law from restating the implementation.
    """
    con = duckdb.connect()
    con.register(FIT, data)
    con.register(THIS, data)
    return con.execute(transform.sql).to_arrow_table()
