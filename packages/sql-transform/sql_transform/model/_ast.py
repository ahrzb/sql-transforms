"""DuckDB as the parser and the printer, and the walk over what it returns.

``json_serialize_sql`` / ``json_deserialize_sql``: one grammar, and it is the
oracle's. The node shapes are internal DuckDB details rather than a documented
API — ``_nodes.py`` types the ones we interpret and ``_shapes.json`` pins the
rest per version, so drift surfaces as a named diff instead of as a wrong
answer.

Nothing here mutates a node. The models are frozen, so a rewrite returns a new
tree and the old one stays valid — which is why the template cache below can
hand the same object to every caller.
"""

import json
from functools import cache
from typing import Any

import duckdb
import pyarrow as pa

from sql_transform.model._errors import TransformError
from sql_transform.model._nodes import (
    AstNode,
    BaseTable,
    Document,
    Function,
    Node,
    Opaque,
    Subquery,
    SubqueryRef,
    TableFunction,
    from_json,
    rebuild,
    to_json,
)

FIT = "__FIT__"


THIS = "__THIS__"


_RECURSIVE_CTE = "RECURSIVE_CTE_NODE"


type Relation = Any  # anything DuckDB will register: arrow, pandas, polars


type Connection = duckdb.DuckDBPyConnection


type LazyRelation = duckdb.DuckDBPyRelation


type Params = dict[str, pa.Table]


type Bindings = dict[str, Relation]


# Everything a transform resolved from the caller's frame, by name: lookup
# relations, foreign transforms, and members. One map so `clone` can carry
# the lot; the two views above are derived from it.
type Captured = dict[str, Any]


def _serialize(sql: str) -> dict[str, Any]:
    (out,) = duckdb.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()
    doc = json.loads(out)
    if doc.get("error"):
        raise TransformError(
            f"parse error at position {doc.get('position')}: {doc.get('error_message')}"
        )
    return doc


def _parse(sql: str) -> Document:
    return from_json(_serialize(sql))


def _deserialize(doc: Document | dict[str, Any]) -> str:
    raw = to_json(doc) if isinstance(doc, AstNode) else doc
    (out,) = duckdb.execute(
        "SELECT json_deserialize_sql(?::JSON)", [json.dumps(raw)]
    ).fetchone()
    return out


def _statement(node: Node) -> Document:
    return Document(error=False, statements=[Subquery(node=node, named_param_map=[])])


@cache
def _template(sql: str) -> Node:
    """A node shape cut from the oracle's own serialization, so a grafted node
    always carries every field the deserializer expects.

    Shared, not copied: the models are frozen, so the old ``json.loads`` of a
    cached string per call bought nothing but garbage.
    """
    return _parse(sql).statements[0].node


def _select_star(name: str) -> Node:
    """A query node reading nothing but ``name``."""
    node = _template("SELECT * FROM __tpl__")
    return node.model_copy(
        update={"from_table": node.from_table.model_copy(update={"table_name": name})}
    )


def _base_table(name: str, alias: str = "") -> BaseTable:
    ref = _template("SELECT * FROM __tpl__").from_table
    return ref.model_copy(update={"table_name": name, "alias": alias})


def _subquery_ref(node: Node, alias: str) -> SubqueryRef:
    ref = _template("SELECT * FROM (SELECT 1) __tpl__").from_table
    return ref.model_copy(
        update={
            "subquery": ref.subquery.model_copy(update={"node": node}),
            "alias": alias,
        }
    )


def _table_function_ref(function: Node, alias: str) -> TableFunction:
    ref = _template("SELECT * FROM range(1)").from_table
    return ref.model_copy(update={"function": function, "alias": alias})


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


def _rename_free[N](body: N, renames: dict[str, str]) -> N:
    """Rename the member's *free* references, not its own definitions.

    A parenthesised derived table already scopes its own CTEs, so the member's
    definitions cannot collide with the caller's. The hazard runs the other
    way: a free name the member resolved from its own frame gets captured by
    an outer CTE that happens to share it. Measured — silent, no error,
    different numbers. The old name survives as the alias, so every column
    reference qualified by it still resolves.
    """

    def rename(node: AstNode) -> AstNode | None:
        if not isinstance(node, BaseTable) or node.table_name not in renames:
            return None
        old = node.table_name
        if renames[old] == old:
            return None  # identity: leave the node exactly as it was
        return node.model_copy(
            update={"table_name": renames[old], "alias": node.alias or old}
        )

    return rebuild(body, rename, deep=True)


def _bind_parameters[N](body: N, args: dict[str, Node]) -> N:
    """Bind the member's two parameters to the call site's arguments.

    An argument that itself mentions ``__FIT__`` — every chained call does —
    is never re-substituted, because ``rebuild`` works bottom-up and never
    offers a replacement back. That used to rest on collecting every site
    before touching any of them.
    """

    def bind(node: AstNode) -> AstNode | None:
        if not isinstance(node, BaseTable) or node.table_name not in args:
            return None
        bound = args[node.table_name]
        return bound.model_copy(update={"alias": node.alias or node.table_name})

    return rebuild(body, bind, deep=True)


def _rename_functions[N](body: N, renames: dict[str, str]) -> N:
    """Rename the member's foreign calls the same way as its free tables, so
    two members can each carry their own ``sc`` without colliding."""

    def rename(node: AstNode) -> AstNode | None:
        if not isinstance(node, Function):
            return None
        stem, _, half = node.function_name.rpartition("_")
        if half in ("fit", "transform") and stem in renames:
            return node.model_copy(update={"function_name": f"{renames[stem]}_{half}"})
        return None

    return rebuild(body, rename, deep=True)


def _list_of(argument: Node) -> Function:
    """``list(arg)``. DuckDB's Python API has no aggregate UDF, so the UDAF
    half is a scalar function over the list DuckDB's own ``list()`` collects.
    The author's text is unchanged; ``GROUP BY`` still does the grouping."""
    call = _template("SELECT list(1)").select_list[0]
    return call.model_copy(update={"children": [argument], "alias": ""})


def _is_recursive_cte(node: Any) -> bool:
    """A ``WITH RECURSIVE`` body. Its self-reference is bound by the enclosing
    entry key rather than by anything inside, which is what stops it being
    hoisted."""
    from sql_transform.model._nodes import RecursiveCte  # noqa: PLC0415

    if isinstance(node, RecursiveCte):
        return True
    return isinstance(node, Opaque) and node.tag_name == _RECURSIVE_CTE
