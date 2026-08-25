"""A projection as a leaf: both halves spliced as SQL, θ as data.

D1 and D2 of `docs/specs/2026-08-11-row-wise-projections-design.md`:

``p_fit(bundle)`` rewrites to a struct of the projection's own aggregates —
grouped by whatever GROUP BY surrounds it, or windowed per-field when the
author wrote ``OVER (...)``, since ``{...} OVER (...)`` itself does not parse.
θ *is* the parameters: a join miss hands ``p_transform`` a NULL θ and every
read of it is NULL, which is P14 falling out of SQL instead of being
implemented.

``p_transform(θ, bundle)`` rewrites to a struct of the residual's output
expressions, with every params read respelled as ``struct_extract`` over θ
and every ``__THIS__`` read respelled as the bundle's own expression for that
column. Nothing is registered, so there is no Python in the row path — and
nothing of the leaf's names survives, which is what makes the splice
capture-free (D3): the refusals below fire at the host's construction and
carry the projection's name.

The leaf-eligible shape, v0: every fit step a plain aggregate SELECT over
``__FIT__``, and a residual whose spine cross-joins one-row params only.
Keys inside the leaf would need θ to carry tables; refused by name.
"""

from dataclasses import dataclass

from sql_transform.model._ast import (
    FIT,
    THIS,
    _aliased,
    _print_expr,
    _template,
    _unaliased,
)
from sql_transform.model._errors import TransformError
from sql_transform.model._nodes import (
    BaseTable,
    ColumnRef,
    Function,
    Join,
    Node,
    Opaque,
    Select,
    cte_entries,
    descendants,
    field,
    is_query,
    rebuild,
)


def _refuse(stem: str, detail: str) -> TransformError:
    return TransformError(f"{stem} is a projection used as a leaf, and {detail}")


@dataclass(frozen=True, slots=True)
class _LeafPlan:
    """The projection, reshaped for splicing: what each half becomes."""

    # (param_name, [(field_name, aggregate expr node)]), in step order
    steps: tuple[tuple[str, tuple[tuple[str, Function], ...]], ...]
    fit_columns: tuple[str, ...]  # __FIT__ columns the steps read, folded
    # (output name, expr node) — the residual's select items
    outputs: tuple[tuple[str, Node], ...]
    this_columns: tuple[str, ...]  # __THIS__ columns the outputs read, folded
    this_alias: str  # folded; how the residual qualifies __THIS__
    params_alias: dict[str, str]  # folded alias -> param table name


def _plain_aggregate_step(
    stem: str, name: str, node: Node
) -> list[tuple[str, Function]]:
    """The step as (field, aggregate) pairs, or the refusal that says why not.

    Windowizing is only sound for a bare aggregate over ``__FIT__``: a WHERE
    or GROUP BY inside the step has no faithful window spelling here yet.
    """
    bad = _refuse(
        stem,
        f"its fit half ({name}) is not a plain aggregate SELECT over {FIT}, "
        "so it has no window spelling — inline that step's shape or serve "
        f"{stem} standalone",
    )
    if not isinstance(node, Select) or cte_entries(node) or node.modifiers:
        raise bad
    if node.where_clause is not None or node.qualify is not None:
        raise bad
    if node.group_expressions or node.group_sets or node.having is not None:
        raise bad
    if not (
        isinstance(node.from_table, BaseTable) and node.from_table.table_name == FIT
    ):
        raise bad
    from sql_transform.model._ast import _aggregates  # noqa: PLC0415

    items: list[tuple[str, Function]] = []
    for item in node.select_list:
        if not (
            isinstance(item, Function)
            and not item.is_operator
            and item.function_name.lower() in _aggregates()
        ):
            raise bad
        field_name = item.alias
        if not field_name:
            raise _refuse(
                stem, f"its fit half ({name}) has an unnamed aggregate — alias it"
            )
        items.append((field_name, item))
    return items


def plan(stem: str, projection) -> _LeafPlan:
    """The leaf plan, or the refusal naming what disqualifies the projection.

    Everything here is static — the projection may be unfitted, and usually
    is: the host's own fit is what will bind ``__FIT__``.
    """
    program = projection._program
    if program.bindings or program.foreign:
        raise _refuse(
            stem,
            "it captures relations or Python leaves of its own, which the "
            "splice cannot carry — a leaf is self-contained SQL over "
            f"{FIT} and {THIS}",
        )
    if any(probe.keys for probe in projection._probes):
        keys = next(p.keys for p in projection._probes if p.keys)
        raise _refuse(
            stem,
            f"it is keyed (a join on ({', '.join(keys)})): a leaf fits per "
            "scope, and θ does not carry keyed tables — serve it standalone, "
            "or let the host's GROUP BY be the key",
        )

    for what, node in (*program.steps, ("residual", program.residual)):
        for v in (node, *descendants(node, deep=True)):
            if isinstance(v, Opaque) and v.fields.get("class") == "LAMBDA":
                raise _refuse(
                    stem,
                    f"its {what} contains a lambda, whose parameter the "
                    "splice cannot tell from a column — serve it standalone",
                )

    steps = []
    fit_columns: set[str] = set()
    for param, node in program.steps:
        items = _plain_aggregate_step(stem, param, node)
        steps.append((param, tuple(items)))
        for _, agg in items:
            for v in (agg, *descendants(agg, deep=True)):
                if isinstance(v, ColumnRef):
                    fit_columns.add(v.column_names[-1].lower())

    from sql_transform.model._projection import _flattened  # noqa: PLC0415

    residual = _flattened(program.residual)
    if not isinstance(residual, Select) or cte_entries(residual):
        raise _refuse(stem, "its residual is not a single SELECT level")
    this_alias = ""
    params_alias: dict[str, str] = {}
    stack: list[Node] = [residual.from_table]
    while stack:
        v = stack.pop()
        match v:
            case BaseTable(table_name=name) if name == THIS:
                this_alias = (v.alias or THIS).lower()
            case BaseTable(table_name=name):
                params_alias[(v.alias or name).lower()] = name
            case Join():
                stack += [v.left, v.right]
            case _:
                raise _refuse(
                    stem,
                    "its residual joins something that is not a one-row params table",
                )

    param_names = {p for p, _ in steps}
    if set(params_alias.values()) - param_names:
        raise _refuse(
            stem, "its residual reads a params table no plain fit step produces"
        )

    outputs: list[tuple[str, Node]] = []
    this_columns: set[str] = set()
    for item in residual.select_list:
        for v in (item, *descendants(item, deep=True)):
            if is_query(v):
                raise _refuse(
                    stem, "its residual nests a subquery, which has no splice"
                )
            if isinstance(v, ColumnRef):
                if len(v.column_names) < 2:
                    raise _refuse(
                        stem,
                        f"the column {v.column_names[0]} in its residual is "
                        "unqualified — qualify it to say which relation it "
                        "reads",
                    )
                qualifier = v.column_names[0].lower()
                if qualifier == this_alias:
                    this_columns.add(v.column_names[-1].lower())
                elif qualifier not in params_alias:
                    raise _refuse(
                        stem,
                        f"{'.'.join(v.column_names)} in its residual reads "
                        "neither the batch nor a params table",
                    )
        name = getattr(item, "alias", "") or (
            item.column_names[-1] if isinstance(item, ColumnRef) else ""
        )
        if not name:
            raise _refuse(stem, "an output of its residual is unnamed — alias it")
        outputs.append((name, item))

    return _LeafPlan(
        steps=tuple(steps),
        fit_columns=tuple(sorted(fit_columns)),
        outputs=tuple(outputs),
        this_columns=tuple(sorted(this_columns)),
        this_alias=this_alias,
        params_alias=params_alias,
    )


_OPS = {"COMPARE_EQUAL": "=", "COMPARE_NOT_DISTINCT_FROM": "IS NOT DISTINCT FROM"}


@dataclass(frozen=True, slots=True)
class _KeyedPlan:
    """A one-grouped-step projection, reshaped for the flat keyed lowering
    (spec M5): its params become plain columns beside the scope keys, and
    the internal join predicate keeps the author's own operator."""

    key_items: tuple[tuple[str, Node], ...]  # (params column name, step expr)
    agg_items: tuple[tuple[str, Function], ...]  # (params column name, aggregate)
    fit_columns: tuple[str, ...]  # __FIT__ columns the step reads, folded
    on_pairs: tuple[tuple[str, str, str], ...]  # (this column, op, params column)
    outputs: tuple[tuple[str, Node], ...]
    this_columns: tuple[str, ...]
    this_alias: str
    params_alias: str


def keyed_plan(stem: str, projection) -> _KeyedPlan:  # noqa: C901
    """The keyed plan, or the refusal naming what disqualifies the projection.

    The admitted shape, v1: exactly one fit step — a grouped SELECT over
    ``__FIT__`` whose non-aggregate items are its group keys — and a residual
    that LEFT JOINs ``__THIS__`` onto that one params table through an
    AND-tree of column equalities.
    """
    program = projection._program
    if program.bindings or program.foreign:
        raise _refuse(
            stem,
            "it captures relations or Python leaves of its own, which the "
            "splice cannot carry — a leaf is self-contained SQL over "
            f"{FIT} and {THIS}",
        )
    for what, node in (*program.steps, ("residual", program.residual)):
        for v in (node, *descendants(node, deep=True)):
            if isinstance(v, Opaque) and v.fields.get("class") == "LAMBDA":
                raise _refuse(
                    stem,
                    f"its {what} contains a lambda, whose parameter the "
                    "splice cannot tell from a column — serve it standalone",
                )

    bad = _refuse(
        stem,
        "its keyed shape is not one grouped step joined once onto "
        f"{THIS} — serve it standalone",
    )
    if len(program.steps) != 1:
        raise bad
    ((param, step),) = program.steps
    if not isinstance(step, Select) or cte_entries(step) or step.modifiers:
        raise bad
    if step.where_clause is not None or step.qualify is not None:
        raise bad
    if step.having is not None or len(step.group_sets or []) > 1:
        raise bad
    if not (
        isinstance(step.from_table, BaseTable) and step.from_table.table_name == FIT
    ):
        raise bad
    if not step.group_expressions:
        raise bad

    from sql_transform.model._ast import _aggregates  # noqa: PLC0415

    group_texts = {_print_expr(g) for g in step.group_expressions}
    key_items: list[tuple[str, Node]] = []
    agg_items: list[tuple[str, Function]] = []
    fit_columns: set[str] = set()
    for item in step.select_list:
        name = getattr(item, "alias", "") or (
            item.column_names[-1] if isinstance(item, ColumnRef) else ""
        )
        if not name:
            raise _refuse(
                stem, f"its fit half ({param}) has an unnamed item — alias it"
            )
        if (
            isinstance(item, Function)
            and not item.is_operator
            and item.function_name.lower() in _aggregates()
        ):
            agg_items.append((name, item))
        elif _print_expr(item) in group_texts:
            key_items.append((name, _unaliased(item)))
        else:
            raise bad
        for v in (item, *descendants(item, deep=True)):
            if isinstance(v, ColumnRef):
                fit_columns.add(v.column_names[-1].lower())

    from sql_transform.model._projection import _flattened  # noqa: PLC0415

    residual = _flattened(program.residual)
    if not isinstance(residual, Select) or cte_entries(residual):
        raise _refuse(stem, "its residual is not a single SELECT level")
    join = residual.from_table
    if not (
        isinstance(join, Join)
        and join.join_type == "LEFT"
        and isinstance(join.left, BaseTable)
        and join.left.table_name == THIS
        and isinstance(join.right, BaseTable)
        and join.right.table_name == param
        and join.condition is not None
        and not join.using_columns
    ):
        raise bad
    this_alias = (join.left.alias or THIS).lower()
    params_alias = (join.right.alias or param).lower()
    columns = {name.lower() for name, _ in (*key_items, *agg_items)}

    on_pairs: list[tuple[str, str, str]] = []
    stack: list[Node] = [join.condition]
    while stack:
        v = stack.pop()
        kind = str(field(v, "type") or "")
        if kind == "CONJUNCTION_AND":
            stack += list(field(v, "children") or [])
            continue
        sides = (field(v, "left"), field(v, "right"))
        if kind not in _OPS or not all(isinstance(s, ColumnRef) for s in sides):
            raise bad
        by = {s.column_names[0].lower(): s.column_names[-1].lower() for s in sides}
        if set(by) != {this_alias, params_alias}:
            raise bad
        if by[params_alias] not in columns:
            raise bad
        on_pairs.append((by[this_alias], _OPS[kind], by[params_alias]))

    outputs: list[tuple[str, Node]] = []
    this_columns: set[str] = {c for c, _, _ in on_pairs}
    for item in residual.select_list:
        for v in (item, *descendants(item, deep=True)):
            if is_query(v):
                raise _refuse(
                    stem, "its residual nests a subquery, which has no splice"
                )
            if isinstance(v, ColumnRef):
                if len(v.column_names) < 2:
                    raise _refuse(
                        stem,
                        f"the column {v.column_names[0]} in its residual is "
                        "unqualified — qualify it to say which relation it "
                        "reads",
                    )
                qualifier = v.column_names[0].lower()
                if qualifier == this_alias:
                    this_columns.add(v.column_names[-1].lower())
                elif qualifier != params_alias:
                    raise _refuse(
                        stem,
                        f"{'.'.join(v.column_names)} in its residual reads "
                        "neither the batch nor a params table",
                    )
                elif v.column_names[-1].lower() not in columns:
                    raise bad
        name = getattr(item, "alias", "") or (
            item.column_names[-1] if isinstance(item, ColumnRef) else ""
        )
        if not name:
            raise _refuse(stem, "an output of its residual is unnamed — alias it")
        outputs.append((name, item))

    return _KeyedPlan(
        key_items=tuple(key_items),
        agg_items=tuple(agg_items),
        fit_columns=tuple(sorted(fit_columns)),
        on_pairs=tuple(on_pairs),
        outputs=tuple(outputs),
        this_columns=tuple(sorted(this_columns)),
        this_alias=this_alias,
        params_alias=params_alias,
    )


def _bundle_fields(
    stem: str, call_children: list[Node], needed: tuple[str, ...], half: str
) -> dict[str, Node]:
    """The struct_pack argument as a name → expression map, checked against
    what the leaf reads."""
    if len(call_children) != 1 or not (
        isinstance(call_children[0], Function)
        and call_children[0].function_name.lower() == "struct_pack"
    ):
        raise _refuse(
            stem,
            f"{stem}_{half} takes one struct_pack(...) bundle naming the "
            "columns it supplies",
        )
    # Unaliased: the field name is struct_pack's business, and it would
    # otherwise ride into the splice as a named argument (`avg(price := x)`).
    fields = {
        child.alias.lower(): _unaliased(child) for child in call_children[0].children
    }
    missing = [c for c in needed if c not in fields]
    if missing:
        raise _refuse(
            stem,
            f"its {half} half reads {', '.join(missing)}, and the bundle "
            f"supplies ({', '.join(sorted(fields)) or 'nothing'})",
        )
    return fields


def _substituted(expr: Node, fields: dict[str, Node]) -> Node:
    """``expr`` with every column reference replaced by the bundle's own
    expression for that column. Shared, not copied: nodes are frozen."""

    def swap(v: Node) -> Node | None:
        if isinstance(v, ColumnRef):
            return fields[v.column_names[-1].lower()]
        return None

    if isinstance(expr, ColumnRef):
        return fields[expr.column_names[-1].lower()]
    return rebuild(expr, swap, deep=True)


def _struct_pack(children: list[Node]) -> Function:
    tpl = _template("SELECT struct_pack(a := 1)").select_list[0]
    return tpl.model_copy(update={"children": children, "alias": ""})


def _extract(inner: Node, key: str) -> Function:
    # key is __param_N or a step field the model named; identifiers only.
    tpl = _template(f"SELECT struct_extract(x, '{key}')").select_list[0]  # noqa: S608
    return tpl.model_copy(update={"children": [inner, tpl.children[1]], "alias": ""})


def fit_call(
    stem: str, projection, children: list[Node], window: Opaque | None
) -> Node:
    """``p_fit(bundle)`` — with the author's window carried onto every field
    when there is one — as the θ struct."""
    leaf = plan(stem, projection)
    fields = _bundle_fields(stem, children, leaf.fit_columns, "fit")
    params: list[Node] = []
    for param, items in leaf.steps:
        packed: list[Node] = []
        for name, agg in items:
            expr: Node = agg.model_copy(
                update={
                    "children": [_substituted(c, fields) for c in agg.children],
                    "alias": "",
                }
            )
            if window is not None:
                # The author's OVER, wholesale — partitions, orders, frame,
                # filter — wearing this aggregate's name and arguments.
                expr = window.model_copy(
                    update={
                        "fields": window.fields
                        | {
                            "function_name": agg.function_name,
                            "children": expr.children,
                            "distinct": agg.distinct,
                            "alias": "",
                        }
                    }
                )
            packed.append(_aliased(expr, name))
        params.append(_aliased(_struct_pack(packed), param))
    return _struct_pack(params)


def bare_call(stem: str, projection, call: Function) -> Node:
    """``p(bundle)`` — the ONE sugar: ``p_transform(p_fit(bundle) OVER (), bundle)``.

    The split spelling is the meaning; the bundle checks, refusals and
    attribution are the halves' own."""
    over = _template("SELECT avg(1) OVER ()").select_list[0]
    theta = fit_call(stem, projection, list(call.children), over)
    return transform_call(
        stem, projection, call.model_copy(update={"children": [theta, *call.children]})
    )


def transform_call(stem: str, projection, call: Function) -> Node:
    """``p_transform(θ, bundle)`` as the residual's outputs over struct reads.

    θ is whatever expression the author passed — a column carrying the fit
    half across a join, or the fit call inlined — and it is read, never
    evaluated twice by us: the node is shared, and deduplication is the
    oracle's own business.
    """
    leaf = plan(stem, projection)
    if len(call.children) != 2:
        raise _refuse(stem, f"{stem}_transform takes (θ, struct_pack(...))")
    theta = call.children[0]
    fields = _bundle_fields(stem, [call.children[1]], leaf.this_columns, "transform")

    def value(ref: ColumnRef) -> Node:
        qualifier = ref.column_names[0].lower()
        column = ref.column_names[-1].lower()
        if qualifier == leaf.this_alias:
            return fields[column]
        param = leaf.params_alias[qualifier]
        return _extract(_extract(theta, param), ref.column_names[-1])

    def swap(v: Node) -> Node | None:
        return value(v) if isinstance(v, ColumnRef) else None

    packed = []
    for name, expr in leaf.outputs:
        replaced = (
            value(expr)
            if isinstance(expr, ColumnRef)
            else rebuild(expr, swap, deep=True)
        )
        packed.append(_aliased(replaced, name))
    return _struct_pack(packed)
