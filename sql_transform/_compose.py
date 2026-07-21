"""Fit-time front-end: inline a fitted SQLTransform referenced via a t-string.

`SQLTransform(t"... {a.transform}(col) ...")` desugars to plain SQL with
`__COMPOSE_i__(col)` placeholder calls plus a ref map; at fit() the placeholders
are replaced by the referenced transform's frozen scalar expression, remapped to
`col` and state name-scoped to `__STATE_R{i}__`. Both frozen (`{a.transform}`)
and unfit (`{a}`) refs are supported, including nesting/chaining and mixing
the two.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from string.templatelib import Template

import datafusion
import pyarrow as pa
import sqlglot
from sqlglot import exp

from sql_transform._rewrite import rewrite_sql
from sql_transform._sql import find_window_aggregates, parse_and_validate
from sql_transform._state import build_state_tables
from sql_transform._transformer_ref import is_transformer


@dataclass(frozen=True)
class Ref:
    transform: object  # a SQLTransform, or a fitted transformer if is_transformer
    frozen: bool  # True for {a.transform}; False for bare {a}
    expr_text: str  # interpolation source, for error messages
    is_transformer: bool = False


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
        elif is_transformer(v):
            ref = Ref(v, frozen=False, expr_text=item.expression, is_transformer=True)
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


def inline_references(
    select: exp.Select,
    refs: dict[str, Ref],
    ctx: datafusion.SessionContext,
    training: pa.Table,
) -> InlineResult:
    """Replace each __COMPOSE_i__(arg) node with the referenced transform's
    frozen, remapped, name-scoped expression. `arg` may itself contain deeper
    __COMPOSE_j__(...) placeholders ({a}({b}(col))); those are resolved first,
    bottom-up, into an inlined expression that becomes the outer ref's input,
    with the inner refs' state cross-joined into the outer's fit. For a frozen
    ref ({a.transform}), inline its already-fitted state. For an unfit ref
    ({a}), fit its DEFINITION into a fresh __STATE_R{i}__ scope first
    (fit_into_scope), then inline that. Mutates `select`. Empty refs -> no-op."""
    scoped_state: dict[str, pa.Table] = {}
    processed: set[str] = set()

    def resolve_arg(node: exp.Anonymous, ref: Ref) -> exp.Expression:
        args = node.expressions
        if len(args) != 1:
            raise ValueError(
                f"{{{ref.expr_text}}} must be applied to a single input "
                "column, e.g. {...}(age)"
            )
        arg = args[0]
        if isinstance(arg, exp.Anonymous) and str(arg.this).upper() in refs:
            return process_ref(str(arg.this).upper(), arg)
        if isinstance(arg, exp.Column):
            return exp.Column(
                this=exp.to_identifier(arg.name, quoted=arg.this.quoted),
                table=exp.to_identifier("__THIS__"),
            )
        raise ValueError(
            f"{{{ref.expr_text}}} must be applied to a single input "
            "column or another reference, e.g. {...}(age) or {...}({...}(age))"
        )

    def process_ref(name: str, node: exp.Anonymous) -> exp.Expression:
        ref = refs[name]
        i = int(name.removeprefix("__COMPOSE_").removesuffix("__"))
        processed.add(name)
        input_expr = resolve_arg(node, ref)  # recurses into deeper placeholders first

        if not ref.frozen:
            if ref.transform._infer_fn is not None:
                raise ValueError(
                    f"{{{ref.expr_text}}} is already fitted -- ambiguous: use "
                    f"{{{ref.expr_text}.transform}} to reuse its frozen state, "
                    f"or reference a fresh unfit instance to re-fit"
                )
            frozen_expr, state = fit_into_scope(
                ref, input_expr, f"__STATE_R{i}__", scoped_state, ctx, training
            )
            scoped_state.update(state)
            return frozen_expr

        _require_frozen_fitted(ref)
        expr, inner_col, scope, state = _frozen_expr(ref.transform, i)

        def rewrite(n: exp.Expression) -> exp.Expression:
            if isinstance(n, exp.Column):
                if n.table == "__THIS__":
                    if n.name == inner_col:
                        return input_expr.copy()
                    return exp.Column(
                        this=exp.to_identifier(n.name, quoted=n.this.quoted),
                        table=exp.to_identifier("__THIS__"),
                    )
                if n.table and n.table.startswith("__STATE"):
                    return exp.Column(
                        this=exp.to_identifier(n.name, quoted=n.this.quoted),
                        table=exp.to_identifier(scope),
                    )
            return n

        scoped_state.update(state)
        return expr.transform(rewrite)

    for name, ref in refs.items():
        if name in processed:
            continue  # already inlined as a nested arg of an outer placeholder
        node = _find_call(select, name, ref)
        node.replace(process_ref(name, node))
    return InlineResult(scoped_state=scoped_state)


def fit_into_scope(
    ref: Ref,
    input_expr: exp.Expression,
    scope: str,
    deeper_states: dict[str, pa.Table],
    ctx: datafusion.SessionContext,
    training: pa.Table,
) -> tuple[exp.Expression, dict[str, pa.Table]]:
    """Fit ref's DEFINITION into `scope`, its input remapped to input_expr,
    cross-joining deeper scopes' states. Returns (frozen_expr, {scope: state}).

    Never calls ref.transform.fit() -- parses ref.transform._sql (the
    definition) fresh, so the referenced transform is left untouched
    (clone contract)."""
    tree = parse_and_validate(ref.transform._sql)
    if len(tree.expressions) != 1:
        raise ValueError("referenced transform must be single-output")
    inner_cols = {c.name for c in tree.find_all(exp.Column)}
    if len(inner_cols) != 1:
        raise ValueError(
            "referenced transform must read exactly one input column "
            "(multi-input not yet supported)"
        )
    inner_col = next(iter(inner_cols))

    # Remap inner's single __THIS__ column -> input_expr throughout the tree,
    # so its window aggregates are over input_expr (agg-over-expression).
    def remap(n):
        if (
            isinstance(n, exp.Column)
            and n.name == inner_col
            and not (n.table and n.table.startswith("__STATE"))
        ):
            return input_expr.copy()
        return n

    tree = tree.transform(remap)

    windows = find_window_aggregates(tree)
    own = build_state_tables(windows, ctx, "__THIS__", join_tables=deeper_states)
    if len(own) > 1:
        raise NotImplementedError(
            "multiple partitioned state tables in a referenced transform are "
            "not supported this slice"
        )
    # Scope: rename ref's produced state tables into `scope` and rewrite refs.
    scoped_state, rename = {}, {}
    for state_name, tbl in own.items():
        rename[state_name] = scope
        scoped_state[scope] = tbl
    frozen = rewrite_sql(tree, windows, extra_marker_tables=())  # str
    frozen_expr = sqlglot.parse_one(frozen).expressions[0]
    if isinstance(frozen_expr, exp.Alias):
        frozen_expr = frozen_expr.this
    # Rebuilt UNQUOTED on purpose, unlike real user columns. These are
    # engine-generated state VALUE-columns, and state_key() builds every one of
    # them lowercase (f"{fn.lower()}_{col.lower()}", e.g. avg_age) -- so the
    # unquoted-identifier folding both engines do (TASK-28) is a no-op here and
    # the ref resolves either way. Do NOT "fix" this by quoting: that would pin a
    # generated name case-exact and silently desync the day state_key's casing
    # changes. Real column identifiers keep the user's own quoting; only these
    # synthesized state keys are safe to rebuild bare.
    frozen_expr = frozen_expr.transform(
        lambda n: (
            exp.column(n.name, table=scope)
            if isinstance(n, exp.Column) and n.table in rename
            else n
        )
    )
    return frozen_expr, scoped_state


def _find_call(select: exp.Select, name: str, ref: Ref) -> exp.Anonymous:
    for n in select.find_all(exp.Anonymous):
        if str(n.this).upper() == name:
            return n
    raise ValueError(
        f"a referenced transform must be applied to a column, "
        f"e.g. {{{ref.expr_text}}}(age)"
    )


def _require_frozen_fitted(ref: Ref) -> None:
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
    # Derive input arity from the inner's ORIGINAL sql, not the rewritten
    # projection: window-aggregate args and PARTITION BY keys are frozen into
    # __STATE__ and vanish from the rewritten projection, undercounting inputs.
    input_cols = {c.name for c in sqlglot.parse_one(inner._sql).find_all(exp.Column)}
    if len(input_cols) != 1:
        raise ValueError(
            "referenced transform must read exactly one input column; "
            "multi-input (incl. PARTITION BY) references are not yet supported"
        )
    inner_col = next(iter(input_cols))
    states = inner._state_tables or {}
    scope = f"__STATE_R{i}__"
    scoped = {scope: next(iter(states.values()))} if states else {}
    return expr, inner_col, scope, scoped
