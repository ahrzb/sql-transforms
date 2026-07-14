"""Rewrite DataFusion logical plans into SQL runnable by the Rust InferFn.

Walks each top-level projection alias and converts its expression tree
into SQL text. Window aggregate references (Alias wrapping a Column, or a
bare Column with a DataFusion-generated window-agg name) are rewritten
into `__STATE__.<fn>_<col>` references; plain columns become
`__THIS__.<col>` references. The result is `SELECT ... FROM __THIS__,
__STATE__` -- a cross join, since __STATE__ is always exactly one row.
"""

from __future__ import annotations

import re

import datafusion
from datafusion.expr import Alias, BinaryExpr, Column

from sql_transform._state import state_key


def rewrite_sql(plan: datafusion.plan.LogicalPlan) -> str:
    """Return a SQL string equivalent to the plan's projection, with every
    window-aggregate reference replaced by a __STATE__ column reference."""
    proj = plan.to_variant()
    parts: list[str] = []

    for raw_p in proj.projections():
        variant = raw_p.to_variant()
        if isinstance(variant, Column):
            out_name = variant.name()
        elif isinstance(variant, Alias):
            out_name = variant.alias()
        else:
            raise ValueError(
                "Expression in SELECT list needs an alias (AS name): "
                f"unsupported top-level expression {type(variant).__name__}"
            )
        if not _VALID_IDENT_RE.match(out_name):
            raise ValueError(
                "Expression in SELECT list needs an explicit alias (AS "
                f"name) -- generated name {out_name!r} is not a valid "
                "identifier"
            )
        # _expr_to_sql unwraps Alias internally, so this single call
        # handles both the Column and Alias cases above identically --
        # only out_name needed the branch.
        expr_sql = _expr_to_sql(raw_p)
        parts.append(f"{expr_sql} AS {out_name}")

    return "SELECT " + ", ".join(parts) + " FROM __THIS__, __STATE__"


def _expr_to_sql(raw_expr) -> str:
    """Convert a RawExpr tree to a SQL expression string."""
    expr = raw_expr.to_variant()

    if isinstance(expr, Column):
        return _column_to_sql(expr.name())

    if isinstance(expr, BinaryExpr):
        left = _expr_to_sql(expr.left())
        right = _expr_to_sql(expr.right())
        return f"({left} {expr.op()} {right})"

    if isinstance(expr, Alias):
        return _expr_to_sql(expr.expr())

    raise ValueError(f"Unrecognized expression node: {type(expr).__name__}")


def _column_to_sql(col_name: str) -> str:
    m = _WINDOW_COL_RE.match(col_name)
    if m:
        key = state_key(m.group("fn"), m.group("col"))
        return f"__STATE__.{key}"
    return f"__THIS__.{col_name}"


# DataFusion generates window aggregate column names like:
#   avg(data.age) ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
_WINDOW_COL_RE = re.compile(r"^(?P<fn>\w+)\((?:\w+\.)?(?P<col>\w+)\)\s")

# A bare SQL identifier -- used to reject DataFusion's raw display text
# (e.g. "mean(data.age)") as an unsafe AS-name when the user didn't supply
# an explicit alias for a window-aggregate projection.
_VALID_IDENT_RE = re.compile(r"^\w+$")
