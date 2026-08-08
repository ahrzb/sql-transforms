"""DuckDB as the parser and the printer, and the walk over what it returns.

``json_serialize_sql`` / ``json_deserialize_sql``: one grammar, and it is the
oracle's. The node shapes below are internal DuckDB details rather than a
documented API — ``_shapes_test.py`` is what keeps them true, and replays
DuckDB's own corpus to catch the format moving.
"""

import copy
import json
from collections.abc import Iterator
from functools import cache
from typing import Any

import duckdb
import pyarrow as pa

from sql_transform.model._errors import TransformError

FIT = "__FIT__"


THIS = "__THIS__"


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


# Everything a transform resolved from the caller's frame, by name: lookup
# relations, foreign transforms, and members. One map so `clone` can carry
# the lot; the two views above are derived from it.
type Captured = dict[str, Any]


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


# `table_macro` as well as `table`: a user's `CREATE MACRO ... AS TABLE` is a
# table function too, and calling one is not a member call.
_TABLE_FUNCTIONS = (
    "SELECT DISTINCT function_name FROM duckdb_functions()"
    " WHERE function_type IN ('table', 'table_macro')"
)


_ALL_FUNCTIONS = "SELECT DISTINCT function_name FROM duckdb_functions()"


@cache
def _default_functions(query: str) -> frozenset[str]:
    """The oracle's own vocabulary, on the default connection. Cached for the
    process because it cannot change: nobody can define a macro there."""
    return frozenset(name.lower() for (name,) in duckdb.execute(query).fetchall())


def _functions(query: str, con: Connection | None) -> frozenset[str]:
    """Every function name in scope.

    Read off ``con`` when there is one. Passing a connection says *use my
    catalog*, and a catalog includes its macros and UDFs — reading the
    module-level default connection instead made the user's own functions
    invisible, so a transform calling one refused at construction while a
    *table* in the same catalog resolved.

    Not cached per connection: that would pin the connection for the life of
    the process, and a macro defined after the first lookup would be missed.
    Construction is not a hot path.
    """
    if con is None:
        return _default_functions(query)
    return frozenset(name.lower() for (name,) in con.execute(query).fetchall())


# Names the connection can *bind*, which is narrower than names it can list.
# Two exclusions, both measured: the 47 internal views on a fresh connection
# (`information_schema.tables`, `duckdb_tables`, ...) are unreachable
# unqualified, and so is a table in an ATTACHed database. Claiming either cost
# the caller their own object of that name — a frame variable called `tables`
# or `columns` is ordinary.
_CATALOG = """
SELECT table_name FROM duckdb_tables()
 WHERE NOT internal AND database_name IN (current_database(), 'temp')
UNION ALL
SELECT view_name FROM duckdb_views()
 WHERE NOT internal AND database_name IN (current_database(), 'temp')
"""


def _catalog(con: Connection | None) -> frozenset[str]:
    """Tables and views the supplied connection already owns.

    Passing a connection is how you say *use my catalog*, so a name it can
    already resolve is not a free reference and must not be looked for in
    the caller's frame — nor renamed when a shared connection is mangled.

    Folded, because the binder is: a connection holding ``Customers`` resolves
    ``customers``, and comparing exact strings both refused that as an unknown
    name and — with a frame object of the folded spelling in reach — let the
    frame quietly answer for the connection's own table.
    """
    if con is None:
        return frozenset()
    return frozenset(name.lower() for (name,) in con.execute(_CATALOG).fetchall())


def normalize(sql: str) -> str:
    """``sql`` as the oracle prints it. The only way to compare SQL text."""
    return _deserialize(_serialize(sql))


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
            if renames[old] == old:
                continue  # identity: leave the node exactly as it was
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


def _list_of(argument: Node) -> Node:
    """``list(arg)``. DuckDB's Python API has no aggregate UDF, so the UDAF
    half is a scalar function over the list DuckDB's own ``list()`` collects.
    The author's text is unchanged; ``GROUP BY`` still does the grouping."""
    call = _template("SELECT list(1)")["select_list"][0]
    call["children"] = [argument]
    call["alias"] = ""
    return call
