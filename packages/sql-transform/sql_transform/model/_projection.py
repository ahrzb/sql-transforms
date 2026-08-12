"""SQLProjection — a row-wise transform, servable one row at a time.

> A projection is a transform whose residual is row-wise: exactly one output
> row per ``__THIS__`` row, computed from that row and the params alone.

Same text, same ``Program`` — plus one gate at construction. The gate runs on
the *residual*, where ``__FIT__`` is already gone: what freezing removed was
never the projection's problem, and what is left over ``__THIS__`` is exactly
what serves. The levels that carry the batch's rows (the *spine*) must be pure
projection over joins; a level that reads only params is free, because it is a
constant table at serving.

Implements `docs/superpowers/specs/2026-08-11-row-wise-projections-design.md`.
"""

import sys
from dataclasses import dataclass, replace
from typing import Any, NoReturn

import pyarrow as pa

from sql_transform.model._analysis import _reads
from sql_transform.model._ast import (
    THIS,
    Captured,
    Connection,
    _aggregates,
    _print_expr,
    _template,
)
from sql_transform.model._errors import NotRowWise, TransformError
from sql_transform.model._nodes import (
    BaseTable,
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
from sql_transform.model._program import Fitted, Program

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
                isinstance(v, Function)
                and not v.is_operator
                and v.function_name.lower() in _aggregates()
            ):
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


@dataclass(slots=True, eq=False, repr=False)
class FittedProjection:
    """``T -> R``, one row out per row in — and the artifact you ship.

    ``params`` is the whole learned state, inspectable. ``transform`` numbers
    the input, runs the ordered residual, and drops the ordinal: SQL results
    are unordered and a params LEFT JOIN really does emit unmatched rows last,
    so input order is threaded through the text, never assumed.
    """

    _fitted: Fitted  # over the *ordered* residual

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
    ) -> None:
        # The frame is read *here*, not inside `compile` — one level deeper
        # would capture from the wrong caller.
        frame = sys._getframe(1)
        scope = frame.f_globals | frame.f_locals
        del frame

        program = Program.compile(sql, scope, connection=connection, captured=captured)
        _refuse_not_row_wise(program.residual)
        self._program = program
        self.connection = program.connection
        self.captured = program.captured
        self.source = program.source
        self.sql = program.sql
        self._ordered = _threaded(program.residual)

    def __repr__(self) -> str:
        return f"SQLProjection({self.sql!r})"

    def fit(self, data: Any) -> FittedProjection:
        """Partial application: the params materialize, the artifact serves."""
        fitted = self._program.fit(data)
        return FittedProjection(replace(fitted, node=self._ordered))

    __call__ = fit
