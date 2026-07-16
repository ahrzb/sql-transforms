"""Fit-time front-end: inline a fitted SQLTransform referenced via a t-string.

`SQLTransform(t"... {a.transform}(col) ...")` desugars to plain SQL with
`__COMPOSE_i__(col)` placeholder calls plus a ref map; at fit() the placeholders
are replaced by the referenced transform's frozen scalar expression, remapped to
`col` and state name-scoped to `__STATE_R{i}__`. Frozen path only.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from string.templatelib import Template

import pyarrow as pa
import sqlglot
from sqlglot import exp


@dataclass(frozen=True)
class Ref:
    transform: object  # a SQLTransform (imported lazily to avoid a cycle)
    frozen: bool       # True for {a.transform}; False for bare {a}
    expr_text: str     # interpolation source, for error messages


@dataclass(frozen=True)
class InlineResult:
    scoped_state: dict[str, pa.Table]


def desugar_template(template: Template) -> tuple[str, dict[str, Ref]]:
    """Turn a t-string into (plain SQL with __COMPOSE_i__ placeholders, ref map)."""
    from sql_transform import SQLTransform

    parts: list[str] = []
    refs: dict[str, Ref] = {}
    i = 0
    for item in template:
        if isinstance(item, str):
            parts.append(item)
            continue
        v = item.value  # Interpolation
        if (
            inspect.ismethod(v)
            and isinstance(v.__self__, SQLTransform)
            and v.__func__ is SQLTransform.transform
        ):
            ref = Ref(v.__self__, frozen=True, expr_text=item.expression)
        elif isinstance(v, SQLTransform):
            ref = Ref(v, frozen=False, expr_text=item.expression)
        else:
            raise TypeError(
                f"interpolation {{{item.expression}}} must be a SQLTransform or "
                f"its .transform, got {type(v).__name__}"
            )
        name = f"__COMPOSE_{i}__"
        refs[name] = ref
        parts.append(name)
        i += 1
    return "".join(parts), refs


def inline_references(select: exp.Select, refs: dict[str, Ref]) -> InlineResult:
    """Replace each __COMPOSE_i__(col) node with the referenced transform's frozen,
    remapped, name-scoped expression. Mutates `select`. Empty refs -> no-op."""
    scoped_state: dict[str, pa.Table] = {}
    for i, (name, ref) in enumerate(refs.items()):
        node = _find_call(select, name, ref)
        argcol = _single_col_arg(node, ref)
        _require_frozen_fitted(ref)
        expr, inner_col, scope, state = _frozen_expr(ref.transform, i)

        # noqa false positive: rewrite() is invoked synchronously via
        # expr.transform() on the next line, within this same loop iteration --
        # argcol/inner_col/scope are never read after the loop advances, so the
        # usual late-binding closure bug doesn't apply here.
        def rewrite(n: exp.Expression) -> exp.Expression:
            if isinstance(n, exp.Column):
                if n.table == "__THIS__":
                    col = argcol if n.name == inner_col else n.name  # noqa: B023
                    return exp.column(col, table="__THIS__")
                if n.table and n.table.startswith("__STATE"):
                    return exp.column(n.name, table=scope)  # noqa: B023
            return n

        node.replace(expr.transform(rewrite))
        scoped_state.update(state)
    return InlineResult(scoped_state=scoped_state)


def _find_call(select: exp.Select, name: str, ref: Ref) -> exp.Anonymous:
    for n in select.find_all(exp.Anonymous):
        if str(n.this).upper() == name:
            return n
    raise ValueError(
        f"a referenced transform must be applied to a column, "
        f"e.g. {{{ref.expr_text}}}(age)"
    )


def _single_col_arg(node: exp.Anonymous, ref: Ref) -> str:
    args = node.expressions
    if len(args) != 1 or not isinstance(args[0], exp.Column):
        raise ValueError(
            f"a referenced transform must be applied to a single input column, "
            f"e.g. {{{ref.expr_text}}}(age)"
        )
    return args[0].name


def _require_frozen_fitted(ref: Ref) -> None:
    if not ref.frozen:
        raise NotImplementedError(
            f"fit-cascade composition ({{{ref.expr_text}}}(col)) is not yet "
            f"implemented; fit it and reference {{{ref.expr_text}.transform}}(col)"
        )
    if ref.transform._infer_fn is None:
        raise ValueError(
            f"referenced transform {{{ref.expr_text}}} is not fitted; "
            f"call .fit(...) before referencing it"
        )


def _frozen_expr(inner, i: int):
    """(expr, inner input col, scope name, {scope: state table}) for a fitted inner."""
    inner_select = sqlglot.parse_one(inner._rewritten_sql)
    if len(inner_select.expressions) != 1:
        raise ValueError(
            "referenced transform must be single-output (one SELECT expression); "
            "multi-output fan-out is not yet supported"
        )
    expr = inner_select.expressions[0]
    if isinstance(expr, exp.Alias):
        expr = expr.this
    this_cols = {c.name for c in expr.find_all(exp.Column) if c.table == "__THIS__"}
    states = inner._state_tables or {}
    if len(this_cols) > 1 or len(states) > 1:
        raise ValueError(
            "referenced transform must read exactly one input column; "
            "multi-input (incl. PARTITION BY) references are not yet supported"
        )
    inner_col = next(iter(this_cols), None)
    scope = f"__STATE_R{i}__"
    scoped = {scope: next(iter(states.values()))} if states else {}
    return expr, inner_col, scope, scoped
