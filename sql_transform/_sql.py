"""Parse and validate SQLTransform's SQL via sqlglot, and locate window
aggregates structurally.

Shared by _state.py and _rewrite.py so there is exactly one place that
knows what SQLTransform's supported SQL subset looks like in the sqlglot
AST -- avoids the class of bug where two independently-maintained regexes
disagreed about the same window aggregate (fixed in commit 2b3171c).
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

_FUNCTION_SYNONYMS = {"MEAN": "AVG"}

_UNSUPPORTED_CLAUSES = {
    "joins": "JOIN",
    "where": "WHERE",
    "group": "GROUP BY",
    "having": "HAVING",
    "order": "ORDER BY",
    "limit": "LIMIT",
}


def parse_and_validate(sql: str) -> exp.Select:
    """Parse `sql` and enforce SQLTransform's supported SQL subset: a
    single SELECT against exactly `FROM __THIS__` (no alias), with none
    of JOIN/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT. Raises ValueError naming
    the first unsupported construct found."""
    statements = sqlglot.parse(sql)
    if len(statements) != 1:
        raise ValueError("Expected exactly one SQL statement")
    tree = statements[0]
    if not isinstance(tree, exp.Select):
        raise ValueError("Only SELECT queries are supported")

    from_ = tree.args.get("from_")
    if from_ is None or not isinstance(from_.this, exp.Table):
        raise ValueError("FROM clause is required and must be a plain table")
    table = from_.this
    if table.name != "__THIS__" or table.alias:
        raise ValueError(
            f"FROM clause must be exactly __THIS__ (no alias); found {table.sql()!r}"
        )

    for key, label in _UNSUPPORTED_CLAUSES.items():
        if tree.args.get(key):
            raise ValueError(f"{label} is not yet supported by SQLTransform")

    return tree


@dataclass(frozen=True)
class WindowAgg:
    """A single window-aggregate reference found in a SELECT list.

    `node` is the actual sqlglot Window node -- rewrite_sql() matches
    against it by identity to know which node to replace, so callers must
    not re-parse the SQL between find_window_aggregates() and using the
    returned WindowAggs.
    """

    node: exp.Window
    fn: str
    col: str
    has_partition: bool
    has_order: bool


def find_window_aggregates(select: exp.Select) -> list[WindowAgg]:
    """Find every window-aggregate node in `select`'s projection list.

    Raises ValueError if a window aggregate's argument isn't a single
    plain column -- multi-arg and expression aggregates aren't supported.
    """
    windows: list[WindowAgg] = []
    for node in select.find_all(exp.Window):
        func = node.this
        if isinstance(func, exp.Anonymous):
            fn = func.this.upper()
            args = func.expressions
        else:
            fn = func.sql_name()
            args = [func.this]
        fn = _FUNCTION_SYNONYMS.get(fn, fn)

        if len(args) != 1 or not isinstance(args[0], exp.Column):
            raise ValueError(
                "Window aggregate argument must be a single plain column: "
                f"{node.sql()!r}"
            )
        col = args[0].name

        windows.append(
            WindowAgg(
                node=node,
                fn=fn,
                col=col,
                has_partition=bool(node.args.get("partition_by")),
                has_order=bool(node.args.get("order")),
            )
        )
    return windows
