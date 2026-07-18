"""Parse and validate SQLTransform's SQL via sqlglot, and locate window
aggregates structurally.

Shared by _state.py and _rewrite.py so there is exactly one place that
knows what SQLTransform's supported SQL subset looks like in the sqlglot
AST -- avoids the class of bug where two independently-maintained regexes
disagreed about the same window aggregate (fixed in commit 2b3171c).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

_FUNCTION_SYNONYMS = {"MEAN": "AVG"}

# Aggregates that take a second argument: the quantile fraction, frozen at fit
# alongside the aggregate itself (e.g. percentile_cont(x, 0.25)). Every other
# supported aggregate is single-argument.
_PARAMETERIZED_AGGS = {"PERCENTILE_CONT", "APPROX_PERCENTILE_CONT"}

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
    arg: exp.Expression  # the aggregate's subject argument (column or expression)
    col: str | None  # arg's column name, or None if arg is an expression
    key: str  # state column name (fn_col / fn_e<hash>; + _p<hash> if params)
    partition_cols: tuple[str, ...]
    has_partition: bool
    has_order: bool
    # extra literal args, e.g. ("0.25",) for percentile_cont(x, 0.25)
    params: tuple[str, ...] = ()


def find_window_aggregates(select: exp.Select) -> list[WindowAgg]:
    """Find every window-aggregate node in `select`'s projection list.

    The subject argument may be a plain column or an arbitrary scalar
    expression. Most aggregates take exactly that one argument; the quantile
    aggregates (percentile_cont / approx_percentile_cont) additionally take a
    literal quantile fraction, frozen alongside the aggregate. Raises ValueError
    for any other multi-argument aggregate.
    """
    windows: list[WindowAgg] = []
    for node in select.find_all(exp.Window):
        func = node.this
        if isinstance(func, exp.Anonymous):
            fn = func.this.upper()
            call_args = list(func.expressions)
        else:
            fn = func.sql_name()
            call_args = [func.this]
            extra = func.args.get("expression")  # typed 2-arg funcs (percentile_cont)
            if extra is not None:
                call_args.append(extra)
        fn = _FUNCTION_SYNONYMS.get(fn, fn)

        if not call_args:
            raise ValueError(f"Window aggregate needs an argument: {node.sql()!r}")
        arg = call_args[0]  # the subject column/expression
        params = call_args[1:]  # extra args (the quantile fraction), if any

        if params and fn not in _PARAMETERIZED_AGGS:
            raise ValueError(
                f"Window aggregate must take exactly one argument: {node.sql()!r}"
            )
        if len(params) > 1:
            raise ValueError(
                f"{fn} takes at most one quantile argument: {node.sql()!r}"
            )
        for p in params:
            # Frozen at fit, so the quantile must be a constant, not a column.
            if not isinstance(p, exp.Literal):
                raise ValueError(
                    f"{fn}'s quantile argument must be a literal: {node.sql()!r}"
                )
        param_sqls = tuple(p.sql() for p in params)

        if isinstance(arg, exp.Column):
            col = arg.name
            key = f"{fn.lower()}_{col.lower()}"  # matches state_key(fn, col)
        else:
            col = None
            digest = hashlib.blake2s(arg.sql().encode(), digest_size=4).hexdigest()
            key = f"{fn.lower()}_e{digest}"
        if param_sqls:
            # Discriminate distinct quantiles of the same column (p25 vs p75).
            joined = ",".join(param_sqls).encode()
            pdigest = hashlib.blake2s(joined, digest_size=4).hexdigest()
            key = f"{key}_p{pdigest}"

        partition_by = node.args.get("partition_by") or []
        partition_cols: list[str] = []
        for p in partition_by:
            if not isinstance(p, exp.Column):
                raise ValueError(
                    f"PARTITION BY must be a list of plain columns: {node.sql()!r}"
                )
            partition_cols.append(p.name)

        windows.append(
            WindowAgg(
                node=node,
                fn=fn,
                arg=arg,
                col=col,
                key=key,
                partition_cols=tuple(partition_cols),
                has_partition=bool(node.args.get("partition_by")),
                has_order=bool(node.args.get("order")),
                params=param_sqls,
            )
        )
    return windows
