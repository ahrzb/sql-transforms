"""SQLProjection — a row-wise transform, servable one row at a time.

> A projection is a transform whose residual is row-wise: exactly one output
> row per ``__THIS__`` row, computed from that row and the params alone.

Same text, same ``Program`` — plus one gate at construction. The gate runs on
the *residual*, where ``__FIT__`` is already gone: what freezing removed was
never the projection's problem, and what is left over ``__THIS__`` is exactly
what serves. The levels that carry the batch's rows (the *spine*) must be pure
projection over joins; a level that reads only params is free, because it is a
constant table at serving.

Implements `docs/specs/2026-08-11-row-wise-projections-design.md`.
"""

import sys
from dataclasses import dataclass, replace
from typing import Any, NoReturn

import duckdb
import pyarrow as pa

from sql_transform.model._analysis import _names_in, _reads
from sql_transform.model._ast import (
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
from sql_transform.model._errors import KeyNotUnique, NotRowWise, TransformError
from sql_transform.model._foreign import _Registry
from sql_transform.model._nodes import (
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
from sql_transform.model._program import Fitted, Program, _lease

# One reason per refused shape. The projection test walks this dict looking
# for gaps, the way the decorrelation test walks its REASONS — a reason
# nothing exercises is a refusal nobody has named.
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
    ``__THIS__`` whole, and must have exactly one row.
    """

    name: str  # the author's name for the side, for the message
    node: Node  # SELECT <keys or *> FROM <side>, CTEs attached, renderable
    keys: tuple[str, ...]  # the author's spelling of each key


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
                )
            )

    walk(residual, {}, [])
    return probes


def _measure(
    probes: list[_Probe], fitted: Fitted, bindings: dict, foreign: dict
) -> None:
    """Run every probe against the materialized params; refuse by name.

    A fresh connection with the artifact's own tables — the same registration
    ``Fitted._bind`` does, minus ``__THIS__``, which no probe reads.
    """
    if not probes:
        return
    con = duckdb.connect()
    try:
        _lease(
            con,
            dict(fitted.params) | dict(bindings),
            foreign,
            _Registry(fitted.instances),
            rename=False,
        )
        for p in probes:
            rel = _deserialize(_statement(p.node))
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
            else:
                (n,) = con.execute(
                    f"SELECT count(*) FROM ({rel}) __cf_probe"  # noqa: S608
                ).fetchone()
                if n != 1:
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
    finally:
        con.close()


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


# The threaded ordinal: harvested from the oracle's own serialization, so the
# grafted nodes carry every field the deserializer expects (P9).
def _row_item() -> Node:
    # ROW is the module's own constant, never user text.
    return _template(f"SELECT {ROW} FROM t").select_list[0]  # noqa: S608


def _row_order() -> list[Node]:
    return list(_template(f"SELECT 1 FROM t ORDER BY {ROW}").modifiers)  # noqa: S608


def _threaded(residual: Node) -> Node:
    """The residual with ``__cf_row`` carried through every spine level and a
    final ORDER BY on it. Every spine level is a plain projection (the gate
    ran first), so the extra select item is always lawful; a level with a
    star already carries the column, because the input table has it.
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
            # ponytail: a spine star always includes __cf_row today; a
            # params-only star (`SELECT p.* FROM __THIS__ t, p`) would lose
            # the thread and refuse loudly at bind rather than serve unordered.
            stars = any(
                isinstance(i, Opaque) and i.fields.get("class") == "STAR"
                for i in node.select_list
            )
            if not stars:
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
    against every column reference (and USING list) in the text; a star keeps
    everything, because a star reads everything.
    """
    parts: set[str] = set()
    for v in (residual, *descendants(residual, deep=True)):
        if isinstance(v, ColumnRef):
            parts.update(p.lower() for p in v.column_names)
        if isinstance(v, Join):
            parts.update(c.lower() for c in v.using_columns)
        if isinstance(v, Opaque) and v.fields.get("class") == "STAR":
            return schema
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


@dataclass(slots=True, eq=False, repr=False)
class FittedProjection:
    """``T -> R``, one row out per row in — and the artifact you ship.

    ``params`` is the whole learned state, inspectable. ``transform`` numbers
    the input, runs the ordered residual, and drops the ordinal: SQL results
    are unordered and a params LEFT JOIN really does emit unmatched rows last,
    so input order is threaded through the text, never assumed.

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
        return self._fitted.params

    @property
    def instances(self) -> dict[int, Any]:
        return self._fitted.instances

    @property
    def sql(self) -> str:
        """The serving text, ordinal and all. What actually executes."""
        return self._fitted.sql

    def transform(self, data: Any) -> pa.Table:
        table = data if isinstance(data, pa.Table) else pa.table(data)
        if ROW in table.column_names:
            raise TransformError(f"input column {ROW} is reserved for the model")
        table = table.append_column(
            ROW, pa.array(range(table.num_rows), type=pa.int64())
        )
        out = self._fitted.transform(table)
        return out.select([i for i, c in enumerate(out.column_names) if c != ROW])

    __call__ = transform

    def compile(self) -> Any:
        """The row path: Confit's ``DuckDBInferFn`` over the same residual and
        the same params, ``shape="map"`` — forced, not chosen, it is the
        scalar-UDF fact seen from the serving side. Confit's contract makes
        this bit-exact with ``transform`` or refuses by name.

        The *unordered* residual: the row path has no batch to reorder, and
        the threaded ordinal would demand a column no serving row has.
        """
        if self._fitted.foreign:
            raise TransformError(
                "a projection calling a Python leaf ("
                + ", ".join(sorted(self._fitted.foreign))
                + ") cannot compile to the row path — there is no Python "
                "there. Serve it in batch with transform(), or wait for "
                "theta-as-data (D1), which makes an SQL leaf of it."
            )
        from confit import DuckDBInferFn  # noqa: PLC0415

        statics = {
            name: table if isinstance(table, pa.Table) else pa.table(table)
            for name, table in {
                **self._fitted.bindings,
                **self._fitted.params,
            }.items()
        }
        return DuckDBInferFn(
            _deserialize(_statement(self._residual)),
            row_tables={THIS: self._row_schema},
            static_tables=statics,
            shape="map",
        )


class SQLProjection:
    """``F -> FittedProjection``: the row-wise sibling of ``SQLTransform``.

    Same two-parameter text, same freezing, same refusals — plus the gate.
    Not a subclass: both classes hold a ``Program``, and neither wants what
    the other adds on top of it.
    """

    def __init__(
        self,
        sql: str,
        connection: Connection | None = None,
        captured: Captured | None = None,
        *,
        _scope: dict[str, Any] | None = None,
    ) -> None:
        # The frame is read *here*, not inside `compile` — one level deeper
        # would capture from the wrong caller. `marginalize` passes the scope
        # it already read instead.
        if _scope is None:
            frame = sys._getframe(1)
            _scope = frame.f_globals | frame.f_locals
            del frame

        program = Program.compile(sql, _scope, connection=connection, captured=captured)
        _refuse_not_row_wise(program.residual)
        self._program = program
        self.connection = program.connection
        self.captured = program.captured
        self.source = program.source
        self.sql = program.sql
        self._ordered = _threaded(program.residual)
        self._probes = _key_probes(program.residual)

    @classmethod
    def marginalize(
        cls,
        sql: str,
        connection: Connection | None = None,
        captured: Captured | None = None,
    ) -> "SQLProjection":
        """The projection a ``__THIS__``-only text means: every fit scope
        (a window aggregate over the spine) frozen over ``__FIT__`` per
        partition and joined back NULL-safe. A rewrite in front of the
        ordinary constructor — one code path below the derived text
        (`docs/specs/2026-08-13-marginalize-design.md`)."""
        frame = sys._getframe(1)
        scope = frame.f_globals | frame.f_locals
        del frame
        from sql_transform.model import _marginal  # noqa: PLC0415

        return cls(_marginal.derive(sql, scope), connection, captured, _scope=scope)

    def __repr__(self) -> str:
        return f"SQLProjection({self.sql!r})"

    def fit(self, data: Any) -> FittedProjection:
        """Partial application: the params materialize, the measurement runs,
        the artifact serves. `KeyNotUnique` fires here — uniqueness is a fact
        about data, the one check construction cannot hoist."""
        table = data if isinstance(data, pa.Table) else pa.table(data)
        fitted = self._program.fit(table)
        _measure(self._probes, fitted, self._program.bindings, self._program.foreign)
        flat = _flattened(self._program.residual)
        return FittedProjection(
            replace(fitted, node=self._ordered),
            flat,
            _serving_schema(_serving_columns(flat, table.schema)),
        )

    __call__ = fit
