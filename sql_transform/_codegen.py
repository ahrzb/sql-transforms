"""Generate pure-Python inference functions from DataFusion logical plans.

Walks each top-level projection alias and converts its expression
tree to Python code. Window aggregate references (Alias wrapping a
Column, or bare Column with DataFusion-generated name) are replaced
with state lookups.
"""

from __future__ import annotations

import re

import datafusion
from datafusion.expr import Alias, BinaryExpr, Column


def generate_infer_fn(
    plan: datafusion.plan.LogicalPlan,
    state: dict,
) -> callable:
    """Return (row: dict) -> dict for single-row inference."""
    proj = plan.to_variant()
    body_lines: list[str] = []

    for raw_p in proj.projections():
        alias = raw_p.to_variant()
        out_name = alias.alias()
        code = _expr_to_python(alias.expr(), state, out_alias=out_name)
        body_lines.append(f'        "{out_name}": {code},')

    source = (
        "def _infer(row, *, _state=None):\n"
        "    return {\n" + "\n".join(body_lines) + "\n    }"
    )
    namespace: dict = {}
    exec(source, {}, namespace)
    fn = namespace["_infer"]

    def bound(row: dict) -> dict:
        return fn(row, _state=state)

    return bound


def _expr_to_python(raw_expr, state: dict, out_alias: str = "") -> str:
    """Convert a RawExpr tree to a Python expression string."""
    expr = raw_expr.to_variant()

    if isinstance(expr, Column):
        col_name = expr.name()
        if _WINDOW_COL_RE.match(col_name):
            val = state.get(out_alias)
            if isinstance(val, dict) and "lookup" in val:
                part_col = val["partition_col"]
                return f'_state[{out_alias!r}]["lookup"][row["{part_col}"]]'
            return f"_state[{out_alias!r}]"
        return f'row["{col_name}"]'

    if isinstance(expr, BinaryExpr):
        left = _expr_to_python(expr.left(), state, out_alias)
        right = _expr_to_python(expr.right(), state, out_alias)
        return f"({left} {expr.op()} {right})"

    if isinstance(expr, Alias):
        inner = expr.expr().to_variant()
        if isinstance(inner, Column):
            val = state.get(out_alias)
            if isinstance(val, dict) and "lookup" in val:
                part_col = val["partition_col"]
                return f'_state[{out_alias!r}]["lookup"][row["{part_col}"]]'
            return f"_state[{out_alias!r}]"
        return _expr_to_python(expr.expr(), state, out_alias)

    return f"None  # unrecognized: {type(expr).__name__}"


# DataFusion generates window aggregate column names like:
#   avg(data.age) ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
_WINDOW_COL_RE = re.compile(r"^\w+\([\w.]+\)\s")
