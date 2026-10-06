"""SQLProjection — a row-wise transform, servable one row at a time.

> A projection is a transform whose residual is row-wise: exactly one output
> row per ``__THIS__`` row, computed from that row and the params alone.

Same text, same ``Program`` — plus one gate at construction. The gate runs on
the *residual*, where ``__FIT__`` is already gone: what freezing removed was
never the projection's problem, and what is left over ``__THIS__`` is exactly
what serves. The levels that carry the batch's rows (the *spine*) must be pure
projection over joins; a level that reads only params is free, because it is a
constant table at serving.

Implements `packages/sql-transform/docs/contract.md`.
"""

import sys
from dataclasses import dataclass, replace
from typing import Any, NoReturn

import pyarrow as pa

from sql_transform._analysis import _bindings_at, _names_in, _reads
from sql_transform._ast import (
    THIS,
    Captured,
    Connection,
    _aggregates,
    _aliased,
    _base_table,
    _deserialize,
    _print_expr,
    _statement,
    _template,
    _unaliased,
)
from sql_transform._errors import KeyNotUnique, NotRowWise, TransformError
from sql_transform._nodes import (
    BaseTable,
    ColumnRef,
    CteEntry,
    Function,
    Join,
    Node,
    Opaque,
    RecursiveCte,
    Select,
    SetOperation,
    SubqueryRef,
    cte_entries,
    descendants,
    field,
    is_query,
    rebuild,
    with_cte_entries,
)
from sql_transform._program import Fitted, Program, _arrow
from sql_transform._udf import PythonTransform, UDFError

# Stable diagnostic keys for the row-local admission refusals.
REASONS: dict[str, str] = {
    "aggregate": "{expr} folds the batch's rows into one value",
    "window": "{expr} reads the batch's other rows through its frame",
    "group-by": "GROUP BY folds the batch's rows together",
    "modifier": "{what} changes which rows come back, or how many",
    "filter": ("{what} drops rows, and a scalar UDF has no encoding for 'no row here'"),
    "this-twice": f"{THIS} enters the row stream {{n}} times, so rows multiply",
    "set-operation": "a set operation stacks the batch onto something else",
    "recursive-cte": "a recursive CTE iterates over the batch",
    "join": "a {what} can drop or duplicate the batch's rows",
    "spine": "{what}",
}

_MODIFIERS = {
    "DISTINCT_MODIFIER": "DISTINCT",
    "ORDER_MODIFIER": "ORDER BY",
    "LIMIT_MODIFIER": "LIMIT",
    "LIMIT_PERCENT_MODIFIER": "LIMIT",
}

ROW = "__cf_row"


def _refuse(reason: str, **fmt: Any) -> NoReturn:
    raise NotRowWise(
        f"a projection serves one output row per {THIS} row, and "
        + REASONS[reason].format(**fmt)
        + " — SQLTransform is the class with no such promise",
        reason,
    )


def _levels(node: Node, reading: dict[str, set[str]]):
    """Every query level under ``node``, with the CTE-reads map in effect
    there. CTE bodies come first, in definition order, exactly as `_plan`
    walks them; nested levels (derived tables, subquery expressions) follow.
    """
    reading = dict(reading)
    for entry in cte_entries(node):
        body = entry.value.query.node
        yield from _levels(body, reading)
        reading[entry.key.lower()] = _reads(body, reading)
    yield node, dict(reading)
    for v in descendants(node, deep=False):
        if is_query(v):
            yield from _levels(v, reading)


def _carries(ref: Node, reading: dict[str, set[str]]) -> bool:
    """Whether this relation reference brings the batch's rows into a level."""
    return THIS in _reads(ref, reading)


def _spine_refs(level: Select, reading: dict[str, set[str]]) -> int:
    """How many times the batch enters this level's row stream: the
    ``__THIS__``-carrying relations directly in its FROM. A carrying derived
    table counts once — its own inside is a level of its own."""
    count = 0
    stack: list[Node] = [level.from_table]
    while stack:
        v = stack.pop()
        match v:
            case BaseTable(table_name=name):
                if name == THIS or THIS in reading.get(name.lower(), set()):
                    count += 1
            case SubqueryRef():
                if _carries(v, reading):
                    count += 1
            case Join():
                stack += [v.left, v.right]
            case _:
                pass
    return count


def _check_joins(level: Select, reading: dict[str, set[str]]) -> None:
    """The batch's side of every join must keep exactly its own rows: LEFT
    with the batch on the left (ASOF included — it matches at most one row),
    RIGHT mirrored, or an unconditional cross join, whose params side is the
    one-row case the fit-time check owns."""
    joins = [
        v
        for v in (level.from_table, *descendants(level.from_table, deep=False))
        if isinstance(v, Join)
    ]
    for j in joins:
        left, right = _carries(j.left, reading), _carries(j.right, reading)
        if not (left or right):
            continue  # params x params: constant at serving, not our rows
        keeps_batch = (
            (j.join_type == "LEFT" and left)
            or (j.join_type == "RIGHT" and right)
            or (
                j.join_type == "INNER"
                and j.ref_type == "CROSS"
                and j.condition is None
                and not j.using_columns
            )
        )
        if not keeps_batch:
            what = f"{j.join_type} join"
            if j.join_type in ("LEFT", "RIGHT"):
                what = f"{j.join_type} join with {THIS} on the dropped side"
            elif j.join_type == "INNER":
                what = "keyed INNER join (a miss drops the row; write LEFT JOIN)"
            _refuse("join", what=what)


def _check_expressions(level: Select, reading: dict[str, set[str]]) -> None:
    """No aggregate, no window, and no ``__THIS__`` on the spine's own select
    list. Nested *params* levels are not descended into — each is a level of
    its own — but one that carries the batch is refused here, at the level
    that embeds it: a subquery expression over ``__THIS__`` reads the batch's
    other rows, whatever its own shape is."""
    for item in level.select_list:
        for v in (item, *descendants(item, deep=False)):
            if is_query(v) and THIS in _reads(v, reading):
                _refuse(
                    "spine",
                    what=f"{THIS} is read from an expression rather than "
                    "FROM, so the value depends on the batch's other rows",
                )
            if isinstance(v, Opaque) and v.fields.get("class") == "WINDOW":
                _refuse("window", expr=_print_expr(v))
            if (
                isinstance(v, Opaque)
                and v.fields.get("class") == "POSITIONAL_REFERENCE"
            ):
                _refuse(
                    "spine",
                    what="a positional reference resolves by position, which "
                    "the model's own appended columns shift",
                )
            if isinstance(v, Function) and not v.is_operator:
                if v.function_name.lower() == "unnest":
                    _refuse(
                        "spine",
                        what=f"{_print_expr(v)} turns one row into none or many",
                    )
                if v.function_name.lower() in _aggregates():
                    _refuse("aggregate", expr=_print_expr(v))


def _refuse_not_row_wise(residual: Node) -> None:
    """The gate: every level that carries the batch's rows is a pure
    projection over row-keeping joins. Levels that read only params are free.
    """
    if THIS not in _reads(residual):
        _refuse(
            "spine",
            what=f"this text never reads {THIS}, so its output cannot track the batch",
        )
    for level, reading in _levels(residual, {}):
        if THIS not in _reads(level, reading):
            continue
        if isinstance(level, SetOperation):
            _refuse("set-operation")
        if isinstance(level, RecursiveCte):
            _refuse("recursive-cte")
        assert isinstance(level, Select)  # query nodes are these three
        for m in level.modifiers:
            kind = str(field(m, "type"))
            _refuse("modifier", what=_MODIFIERS.get(kind, kind))
        if level.where_clause is not None:
            _refuse("filter", what="WHERE")
        if level.qualify is not None:
            _refuse("filter", what="QUALIFY")
        if (
            level.group_expressions
            or level.group_sets
            or level.having is not None
            or level.aggregate_handling != "STANDARD_HANDLING"
        ):
            _refuse("group-by")
        refs = _spine_refs(level, reading)
        if refs == 0:
            _refuse(
                "spine",
                what=f"{THIS} is read from an expression rather than FROM, "
                "so the output rows are not the batch's rows",
            )
        if refs > 1:
            _refuse("this-twice", n=refs)
        _check_joins(level, reading)
        _check_expressions(level, reading)


_EQUALITIES = ("COMPARE_EQUAL", "COMPARE_NOT_DISTINCT_FROM")


@dataclass(frozen=True, slots=True)
class _Probe:
    """One join the fit-time measurement has to clear.

    ``keys`` are the joined side's columns from the equality conjuncts —
    unique keys bound the matches at one, and extra non-equality conjuncts
    only filter further. No keys at all means the relation sits beside
    ``__THIS__`` whole: an ``outer`` side (LEFT/RIGHT, the batch preserved)
    may hold zero rows or one, a CROSS side exactly one — its miss would drop
    the row.
    """

    name: str  # the author's name for the side, for the message
    node: Node  # SELECT <keys or *> FROM <side>, CTEs attached, renderable
    keys: tuple[str, ...]  # the author's spelling of each key
    outer: bool  # the batch side is preserved: a miss is a NULL, not a drop


def _eq_keys(condition: Node, side_names: set[str]) -> list[ColumnRef] | None:
    """The joined side's columns from the AND-tree of equality conjuncts.

    ``None`` means the condition has a shape uniqueness cannot reason about —
    an OR — so the caller falls back to the one-row rule. Conjuncts that are
    not side-keyed equalities are ignored: they only filter matches, and a
    LEFT join's filtered miss keeps the row.
    """
    keys: list[ColumnRef] = []
    stack = [condition]
    while stack:
        v = stack.pop()
        kind, cls = field(v, "type"), field(v, "class")
        if cls == "CONJUNCTION":
            if kind != "CONJUNCTION_AND":
                return None
            stack += list(field(v, "children") or [])
            continue
        if cls == "COMPARISON" and kind in _EQUALITIES:
            for a, b in (
                (field(v, "left"), field(v, "right")),
                (field(v, "right"), field(v, "left")),
            ):
                mine = (
                    isinstance(a, ColumnRef)
                    and len(a.column_names) >= 2
                    and a.column_names[0].lower() in side_names
                )
                other_mine = (
                    isinstance(b, ColumnRef)
                    and len(b.column_names) >= 2
                    and b.column_names[0].lower() in side_names
                )
                if mine and not other_mine:
                    keys.append(a)
                    break
    return keys


def _side_names(side: Node) -> set[str]:
    names = _names_in(side)
    for f in ("alias", "table_name"):
        if value := field(side, f):
            names.add(str(value).lower())
    return names


def _probe_node(side: Node, keys: list[ColumnRef], ctes: list[CteEntry]) -> Node:
    """``SELECT <keys> FROM <side>`` with every CTE in scope attached, so a
    side that names one still resolves when rendered standalone."""
    template = _template("SELECT * FROM __tpl__")
    items: list[Node] = list(template.select_list)
    if keys:
        items = [k.model_copy(update={"alias": f"__k{i}"}) for i, k in enumerate(keys)]
    node = template.model_copy(update={"select_list": items, "from_table": side})
    return with_cte_entries(node, ctes)


def _key_probes(residual: Node) -> list[_Probe]:
    """Every spine join, as the measurement fit has to run.

    ASOF is exempt by its own semantics: it matches at most one row per probe
    row whatever the side holds.
    """
    probes: list[_Probe] = []

    def walk(node: Node, reading: dict[str, set[str]], ctes: list[CteEntry]) -> None:
        reading = dict(reading)
        ctes = list(ctes)
        for entry in cte_entries(node):
            walk(entry.value.query.node, reading, ctes)
            reading[entry.key.lower()] = _reads(entry.value.query.node, reading)
            ctes.append(entry)
        for v in descendants(node, deep=False):
            if is_query(v):
                walk(v, reading, ctes)
        if not isinstance(node, Select) or THIS not in _reads(node, reading):
            return
        joins = [
            v
            for v in (node.from_table, *descendants(node.from_table, deep=False))
            if isinstance(v, Join)
        ]
        for j in joins:
            left = THIS in _reads(j.left, reading)
            right = THIS in _reads(j.right, reading)
            if left == right or j.ref_type == "ASOF":
                continue  # params x params, or a join that matches at most one
            side = j.right if left else j.left
            if j.using_columns:
                keys: list[ColumnRef] | None = [
                    ColumnRef.model_construct(
                        class_="COLUMN_REF",
                        type="COLUMN_REF",
                        alias="",
                        query_location=0,
                        column_names=[c],
                    )
                    for c in j.using_columns
                ]
            elif j.condition is not None:
                keys = _eq_keys(j.condition, _side_names(side))
            else:
                keys = []
            keys = keys or []  # an OR condition proves nothing: one-row rule
            name = str(field(side, "alias") or field(side, "table_name") or "")
            probes.append(
                _Probe(
                    name=name or "the joined relation",
                    node=_probe_node(side, keys, ctes),
                    keys=tuple(".".join(k.column_names) for k in keys),
                    outer=j.join_type in ("LEFT", "RIGHT"),
                )
            )

    walk(residual, {}, [])
    return probes


def _measure(probes: list[_Probe], fitted: Fitted) -> None:
    """Run every probe against the artifact's own params; refuse by name.

    The same lease batch execution takes — the caller's connection when there
    is one, so a catalog relation the residual joins is measured where it
    binds — minus ``__THIS__``, which no probe reads.
    """
    if not probes:
        return
    with fitted._leased() as (con, render, _):
        for p in probes:
            rel = render(p.node)
            if p.keys:
                cols = ", ".join(f"__k{i}" for i in range(len(p.keys)))
                # The key tiebreak keeps the refusal deterministic: two keys
                # tied on count made the message flap between runs.
                hit = con.execute(
                    f"SELECT {cols}, count(*) AS n FROM ({rel}) __cf_probe "  # noqa: S608
                    f"GROUP BY {cols} HAVING count(*) > 1 "
                    f"ORDER BY n DESC, {cols} LIMIT 1"
                ).fetchone()
                if hit:
                    *values, n = hit
                    shown = ", ".join(
                        f"{k} = {v!r}" for k, v in zip(p.keys, values, strict=True)
                    )
                    raise KeyNotUnique(
                        f"{p.name} joins {THIS} on ({', '.join(p.keys)}), but "
                        f"({shown}) has {n} rows, so one serving row would "
                        f"become {n}. Aggregate or de-duplicate it."
                    )
                continue
            (n,) = con.execute(
                f"SELECT count(*) FROM ({rel}) __cf_probe"  # noqa: S608
            ).fetchone()
            if n == 1 or (p.outer and n == 0):
                continue
            if p.outer:
                raise KeyNotUnique(
                    f"{p.name} joins {THIS} with no join key and has {n} rows, "
                    f"so one serving row would become {n}. Aggregate it to "
                    "one row, or join it on a key."
                )
            became = (
                f"one serving row would become {n}"
                if n
                else "every serving row would disappear"
            )
            raise KeyNotUnique(
                f"{p.name} sits beside {THIS} with no join key and has "
                f"{n} rows, so {became}. Aggregate it to one row, or "
                "join it on a key."
            )


def _passthrough(node: Node) -> str | None:
    """The base table behind ``SELECT * FROM <base>`` — the exact shape
    freezing synthesizes for every frozen subtree — or None."""
    if not isinstance(node, Select) or cte_entries(node) or node.modifiers:
        return None
    if node.where_clause is not None or node.qualify is not None:
        return None
    if node.group_expressions or node.group_sets or node.having is not None:
        return None
    if not isinstance(node.from_table, BaseTable):
        return None
    if len(node.select_list) != 1:
        return None
    (item,) = node.select_list
    bare_star = (
        isinstance(item, Opaque)
        and item.fields.get("class") == "STAR"
        and not field(item, "relation_name")
        and not field(item, "exclude_list")
        and not field(item, "replace_list")
    )
    return node.from_table.table_name if bare_star else None


def _on_true() -> Node:
    # `1 = 1`, not `TRUE`: the literal round-trips through the printer as
    # CAST('t' AS BOOLEAN), a cast the row path's vocabulary refuses.
    return _template("SELECT 1 FROM a LEFT JOIN b ON 1 = 1").from_table.condition


def _flattened(
    node: Node,
    renames: dict[str, str] | None = None,
    reading: dict[str, set[str]] | None = None,
) -> Node:
    """The residual, respelled for the row path. Three rewrites, all no-ops
    to DuckDB and load-bearing to Confit's stricter surface.

    Freezing's own passthroughs inline: ``(SELECT * FROM __param_0) f`` says
    nothing ``__param_0 AS f`` does not, and Confit's FROM takes tables and
    joins, not derived tables. Author-written params subqueries that do more
    than pass through stay — Confit refuses those loudly by its own name.

    A cross join beside the batch becomes ``LEFT JOIN ... ON TRUE``: Confit's
    map shape statically refuses an INNER join (a miss drops rows), and it
    cannot know what the fit-time probe measured — that the params side is
    exactly one row, which makes the two spellings the same relation."""
    renames = dict(renames or {})
    reading = dict(reading or {})
    kept = []
    for entry in cte_entries(node):
        body = _flattened(entry.value.query.node, renames, reading)
        reading[entry.key.lower()] = _reads(body, reading)
        if base := _passthrough(body):
            renames[entry.key.lower()] = base
            continue
        kept.append(
            entry.model_copy(
                update={
                    "value": entry.value.model_copy(
                        update={
                            "query": entry.value.query.model_copy(update={"node": body})
                        }
                    )
                }
            )
        )
    node = with_cte_entries(node, kept)
    node = rebuild(
        node,
        lambda v: _flattened(v, renames, reading) if is_query(v) else None,
        deep=False,
    )

    def flat_ref(v: Node) -> Node | None:
        match v:
            case SubqueryRef() if base := _passthrough(v.subquery.node):
                return _base_table(base, v.alias)
            case BaseTable(table_name=name) if name.lower() in renames:
                # The author's name stays as the alias, so `s.store` still
                # resolves after `s` becomes `__param_s`.
                return _base_table(renames[name.lower()], v.alias or name)
        return None

    node = rebuild(node, flat_ref, deep=False)

    def cross_to_left(v: Node) -> Node | None:
        if (
            isinstance(v, Join)
            and v.join_type == "INNER"
            and v.ref_type == "CROSS"
            and v.condition is None
            and not v.using_columns
        ):
            left = THIS in _reads(v.left, reading)
            right = THIS in _reads(v.right, reading)
            if left != right:
                this_side, one_row = (v.left, v.right) if left else (v.right, v.left)
                return v.model_copy(
                    update={
                        "join_type": "LEFT",
                        "ref_type": "REGULAR",
                        "condition": _on_true(),
                        "left": this_side,
                        "right": one_row,
                    }
                )
        return None

    node = rebuild(node, cross_to_left, deep=False)

    def unwrap_extract(v: Node) -> Node | None:
        # A field read of a struct_pack — `struct_extract(struct_pack(k :=
        # e, ...), 'k')` or the `.k` operator spelling (measured: an Opaque,
        # class OPERATOR, type STRUCT_EXTRACT) — becomes the field's own
        # expression: struct_pack is pure, so this is a no-op to DuckDB, and
        # it is exactly what the leaf splice writes for a field-addressed
        # output, in a named-argument form Confit's row path refuses.
        if (
            isinstance(v, Function)
            and v.function_name.lower() == "struct_extract"
            and len(v.children) == 2
        ):
            pack, key = v.children
        elif (
            isinstance(v, Opaque)
            and v.fields.get("class") == "OPERATOR"
            and v.fields.get("type") == "STRUCT_EXTRACT"
            and len(v.fields.get("children") or []) == 2
        ):
            pack, key = v.fields["children"]
        else:
            return None
        if not (
            isinstance(pack, Function) and pack.function_name.lower() == "struct_pack"
        ):
            return None
        name = _constant_text(key)
        if name is None:
            return None
        for child in pack.children:
            if str(field(child, "alias") or "").lower() == name.lower():
                expr = _unaliased(child)
                alias = str(field(v, "alias") or "")
                return _aliased(expr, alias) if alias else expr
        return None

    return rebuild(node, unwrap_extract, deep=True)


def _constant_text(v: Node) -> str | None:
    """The string a VALUE_CONSTANT carries, or None for any other shape."""
    if not (isinstance(v, Opaque) and v.fields.get("class") == "CONSTANT"):
        return None
    inner = v.fields.get("value")
    if isinstance(inner, Opaque) and isinstance(inner.fields.get("value"), str):
        return inner.fields["value"]
    return None


# Build ordinal nodes through the oracle so every serialized AST field is present.
def _row_item() -> Node:
    # ROW is the module's own constant, never user text.
    return _template(f"SELECT {ROW} FROM t").select_list[0]  # noqa: S608


def _row_order() -> list[Node]:
    return list(_template(f"SELECT 1 FROM t ORDER BY {ROW}").modifiers)  # noqa: S608


def _threaded(residual: Node) -> Node:
    """The residual with ``__cf_row`` carried through every spine level and a
    final ORDER BY on it. Every spine level is a plain projection (the gate
    ran first), so the extra select item is always lawful. A star over the
    batch's side already carries the column — the input has it — while a
    star qualified by a params side does not, so that level gets it appended.
    """

    def thread(node: Node, reading: dict[str, set[str]]) -> Node:
        reading = dict(reading)
        entries = []
        for entry in cte_entries(node):
            body = thread(entry.value.query.node, reading)
            reading[entry.key.lower()] = _reads(body, reading)
            entries.append(
                entry.model_copy(
                    update={
                        "value": entry.value.model_copy(
                            update={
                                "query": entry.value.query.model_copy(
                                    update={"node": body}
                                )
                            }
                        )
                    }
                )
            )
        node = with_cte_entries(node, entries)
        node = rebuild(
            node, lambda v: thread(v, reading) if is_query(v) else None, deep=False
        )
        if isinstance(node, Select) and THIS in _reads(node, reading):
            bound = _bindings_at(node, reading)
            carried = any(
                isinstance(i, Opaque)
                and i.fields.get("class") == "STAR"
                and (
                    not (rel := field(i, "relation_name"))
                    or bound.get(str(rel).lower(), False)
                )
                for i in node.select_list
            )
            if not carried:
                node = node.model_copy(
                    update={"select_list": [*node.select_list, _row_item()]}
                )
        return node

    ordered = thread(residual, {})
    return ordered.model_copy(update={"modifiers": _row_order()})


def _serving_columns(residual: Node, schema: pa.Schema) -> pa.Schema:
    """The fit columns the residual can actually read — the serving contract.

    A label column nothing references must not be in the row model at all:
    Confit requires every declared attribute on every input row, so keeping it
    would make serving demand a column training never served. Kept by name
    against every column reference (and USING list) in the text. Only a star
    over a request-carrying relation keeps the full request schema.
    """
    for level, reading in _levels(residual, {}):
        if not isinstance(level, Select) or THIS not in _reads(level, reading):
            continue
        bound = _bindings_at(level, reading)
        for item in level.select_list:
            if isinstance(item, Opaque) and item.fields.get("class") == "STAR":
                relation = str(field(item, "relation_name") or "").lower()
                if not relation or bound.get(relation, False):
                    return schema
    parts: set[str] = set()
    for v in (residual, *descendants(residual, deep=True)):
        if isinstance(v, ColumnRef):
            parts.update(p.lower() for p in v.column_names)
        if isinstance(v, Join):
            parts.update(c.lower() for c in v.using_columns)
    kept = [f for f in schema if f.name.lower() in parts]
    return pa.schema(kept)


def _serving_schema(schema: pa.Schema) -> pa.Schema:
    """The serving row schema, derived from the fit relation's schema.

    Every field is nullable (serving rows may carry NULLs the fit data never
    did), widths are real (an int32 fit column binds INTEGER on the row
    path), and out-of-vocabulary types pass through unchanged: Confit keeps
    them opaque unless the SQL references them.
    """
    return pa.schema([pa.field(f.name, f.type) for f in schema])


def _free_tables(node: Node, defined: frozenset[str] = frozenset()) -> set[str]:
    """The base tables ``node`` names that no CTE in scope defines — the
    relations something outside the text has to supply."""
    defined = defined | {e.key.lower() for e in cte_entries(node)}
    free: set[str] = set()
    for entry in cte_entries(node):
        free |= _free_tables(entry.value.query.node, defined)
    for v in descendants(node, deep=False):
        if is_query(v):
            free |= _free_tables(v, defined)
        elif isinstance(v, BaseTable) and v.table_name.lower() not in defined:
            free.add(v.table_name)
    return free


@dataclass(slots=True, eq=False, repr=False)
class FittedProjection:
    """``T -> R``, one row out per row in — and the artifact you ship.

    For Confit-servable projections, ``sql``, ``schema``, ``params`` and ``udfs``
    are the complete public serving artifact. ``params`` is one stored mapping:
    captured statics normalized to Arrow once at fit, plus learned tables.
    Batch, probes and ``compile`` all read that same dict. Arrow buffers remain
    caller-owned and must not be mutated.

    ``transform`` returns Arrow in input order; the public ``sql`` is unordered.

    ``compile`` hands back Confit's own serving function, unwrapped — its
    surface is not re-exported here, and a fresh object per call means no
    cached function for a refit to remember to invalidate.
    """

    _fitted: Fitted  # over the *ordered* residual
    _residual: Node  # the unordered residual: what the row path executes
    _row_schema: pa.Schema  # derived from the fit relation's schema

    def __repr__(self) -> str:
        return f"FittedProjection({self._fitted!r})"

    @property
    def params(self) -> dict[str, pa.Table]:
        """The stored static-table mapping, shared by batch and serving."""
        return self._fitted.params

    @property
    def instances(self) -> dict[int, Any]:
        """Opaque-ID instance state belonging to this artifact's params and UDFs."""
        return self._fitted.instances

    @property
    def udfs(self) -> dict[str, Any]:
        """Runtime UDF views under the unleased names used by public serving SQL."""
        return self._fitted.udfs

    @property
    def sql(self) -> str:
        """The standalone serving query: unordered, one row per request row,
        under the names ``params`` and ``udfs`` use."""
        return _deserialize(_statement(self._residual))

    @property
    def schema(self) -> pa.Schema:
        """The request row schema: the fit columns the query reads, nullable."""
        return self._row_schema

    def transform(self, data: Any) -> pa.Table:
        """Apply eagerly in request order; ``__cf_row`` input names are reserved."""
        # LEFT joins need not preserve input order. The private query threads
        # an ordinal through the spine; it is removed from the returned table.
        table = _arrow(data)
        if any(c.lower() == ROW for c in table.column_names):
            raise TransformError(f"input column {ROW} is reserved for the model")
        table = table.append_column(
            ROW, pa.array(range(table.num_rows), type=pa.int64())
        )
        out = self._fitted.transform(table)
        return out.select([i for i, c in enumerate(out.column_names) if c != ROW])

    __call__ = transform

    def compile(self) -> Any:
        """The row path: Confit's ``DuckDBInferFn`` over the public fields,
        ``shape="map"`` — forced, not chosen, it is the scalar-UDF fact seen
        from the serving side. Confit's contract makes this bit-exact with
        ``transform`` or refuses by name.
        """
        if self._fitted.foreign:
            raise TransformError(
                "a projection calling relation-batch callbacks ("
                + ", ".join(sorted(self._fitted.foreign))
                + ") cannot compile to the row path. "
                "Serve it in batch with transform()."
            )
        known = {name.lower() for name in self.params} | {THIS.lower()}
        catalog = sorted(
            n for n in _free_tables(self._residual) if n.lower() not in known
        )
        if catalog:
            raise TransformError(
                f"{', '.join(catalog)} binds from the caller's connection "
                "catalog, which the row path does not have. Serve it in batch "
                "with transform(), or capture an Arrow snapshot "
                "(captured={name: table}) and refit."
            )
        from confit import DuckDBInferFn  # noqa: PLC0415

        return DuckDBInferFn(
            self.sql,
            row_tables={THIS: self.schema},
            static_tables=self.params,
            udfs=list(self.udfs.values()),
            shape="map",
        )


class SQLProjection:
    """Compile authored FIT/THIS SQL into a row-local fitted transform.

    Construction checks the residual's structural row locality; fit checks
    params cardinality. Confit compilation remains a separate admission check.
    Use ``marginalize`` only for the bounded THIS-only window convenience.

    ``source`` with ``captured`` is the replay input; ``sql`` is resolved
    diagnostic SQL. Explicit captures override caller-frame names and retain
    their mapping identity (see ``Program.compile``). ``connection`` is borrowed
    for fit, probes and batch, not exported to Confit serving.
    """

    def __init__(
        self,
        sql: str,
        connection: Connection | None = None,
        captured: Captured | None = None,
        *,
        _scope: dict[str, Any] | None = None,
    ) -> None:
        # See _program's module contract: read this caller's scope, or reuse
        # the original scope supplied by marginalize.
        if _scope is None:
            frame = sys._getframe(1)
            _scope = frame.f_globals | frame.f_locals
            del frame

        program = Program.compile(
            sql, _scope, connection=connection, captured=captured, row_udfs=True
        )
        _refuse_not_row_wise(program.residual)
        self._program = program
        self.connection = program.connection  # borrowed; None uses owned connections
        self.captured = program.captured  # adopted author mapping for source replay
        # Authored explicit SQL, including derived marginal SQL.
        self.source = program.source
        self.sql = program.sql  # resolved diagnostics, not replay input
        self._ordered = _threaded(program.residual)
        self._probes = _key_probes(program.residual)

    @classmethod
    def marginalize(
        cls,
        sql: str,
        connection: Connection | None = None,
        captured: Captured | None = None,
    ) -> "SQLProjection":
        """Derive explicit fit/request SQL from one admitted THIS-only SELECT.

        Admitted window values are frozen over FIT and joined back NULL-safe.
        Unsupported shapes raise ``TransformError``; write their fit/request
        stages explicitly instead. See the authoring contract for admission.
        """
        frame = sys._getframe(1)
        scope = frame.f_globals | frame.f_locals
        del frame
        from sql_transform import _marginal  # noqa: PLC0415

        return cls(
            _marginal.derive(sql, scope | (captured or {})),
            connection,
            captured,
            _scope=scope,
        )

    def __repr__(self) -> str:
        return f"SQLProjection({self.sql!r})"

    def fit(self, data: Any) -> FittedProjection:
        """Learn params and UDF schemas, then check join cardinality.

        Raises ``KeyNotUnique`` for fan-out or an empty CROSS side. Binding and
        learned-estimator schema refusals also depend on fit data.
        """
        table = _arrow(data)
        fitted = self._program.fit(table)
        for value in (
            self._program.residual,
            *descendants(self._program.residual, deep=True),
        ):
            if (
                isinstance(value, Function)
                and value.function_name.lower() == "struct_extract"
            ):
                children = value.children
            elif (
                isinstance(value, Opaque)
                and field(value, "class") == "OPERATOR"
                and field(value, "type") == "STRUCT_EXTRACT"
            ):
                children = field(value, "children") or []
            else:
                continue
            if len(children) != 2 or not isinstance(children[0], Function):
                continue
            udf = fitted.udfs.get(children[0].function_name)
            name = _constant_text(children[1])
            if isinstance(udf, PythonTransform) and name is not None:
                try:
                    udf.lane_of(name)
                except UDFError as exc:
                    raise TransformError(str(exc)) from exc
        _measure(self._probes, fitted)
        raw = {
            descriptor.udf_name: fitted.udfs[descriptor.udf_name]
            for descriptor in self._program.estimators.values()
        }
        sql_types = {
            pa.int64(): "BIGINT",
            pa.float64(): "DOUBLE",
            pa.bool_(): "BOOLEAN",
            pa.string(): "VARCHAR",
        }
        casts = {
            dtype: _template(f"SELECT CAST(1 AS {sql_types[dtype]})").select_list[0]
            for udf in raw.values()
            for dtype in udf.takes.types
        }

        def typed_raw(value: Node) -> Node | None:
            if not isinstance(value, Function) or value.function_name not in raw:
                return None
            udf = raw[value.function_name]
            # Make DuckDB's declared-UDF argument coercion explicit for Confit.
            children = [value.children[0]]
            for argument, dtype in zip(
                value.children[1:], udf.takes.types, strict=True
            ):
                cast = casts[dtype]
                children.append(
                    cast.model_copy(
                        update={
                            "fields": cast.fields | {"child": _unaliased(argument)},
                        }
                    )
                )
            return value.model_copy(update={"children": children})

        flat = _flattened(rebuild(self._program.residual, typed_raw, deep=True))
        return FittedProjection(
            replace(fitted, node=rebuild(self._ordered, typed_raw, deep=True)),
            flat,
            _serving_schema(_serving_columns(flat, table.schema)),
        )

    __call__ = fit
