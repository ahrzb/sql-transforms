"""Rewrite SQLTransform's SQL into LEFT-joined state-table SQL for the engines.

Each window-aggregate reference becomes a column reference into its partition's
state table (state_table_name(partition_cols).state_key(fn, col)); every plain
column becomes __THIS__.col; and one LEFT JOIN per distinct partition-key-set is
appended. The global OVER () state (empty key-set) is joined on a constant marker
key. All joins are LEFT onto unique-keyed (GROUP BY) tables, so the rewrite is
strictly 1-to-1: an unseen key yields NULL, never a dropped or duplicated row.
"""

from __future__ import annotations

from sqlglot import exp

from sql_transform._sql import WindowAgg
from sql_transform._state import STATE_MARKER, state_key, state_table_name


def rewrite_sql(
    select: exp.Select,
    windows: list[WindowAgg],
    extra_marker_tables: tuple[str, ...] = (),
) -> str:
    """Return SQL equivalent to `select` with window aggregates replaced by
    state-table column references and one LEFT JOIN per partition-key-set.

    Mutates `select` in place -- callers should not reuse it afterwards."""
    window_ref = {
        id(w.node): (
            state_table_name(w.partition_cols),
            state_key(w.fn, w.col),
            w.partition_cols,
        )
        for w in windows
    }

    # One LEFT JOIN per distinct partition-key-set, in first-seen order as the
    # SELECT list is scanned left to right (not `windows`' own order, since
    # find_all(exp.Window) doesn't guarantee SELECT-list order).
    seen: dict[tuple[str, ...], None] = {}

    for e in select.expressions:
        out_name = e.alias_or_name
        if not out_name:
            raise ValueError(
                f"Expression in SELECT list needs an alias (AS name): {e.sql()!r}"
            )

        for win_node in list(e.find_all(exp.Window)):
            table, col, partition_cols = window_ref[id(win_node)]
            seen.setdefault(partition_cols, None)
            win_node.replace(exp.column(col, table=table))

        for col_node in list(e.find_all(exp.Column)):
            if col_node.table and col_node.table.startswith("__STATE"):
                continue  # already rewritten above
            if col_node.table and col_node.table != "__THIS__":
                raise ValueError(
                    f"Column qualifier {col_node.table!r} does not refer to __THIS__"
                )
            col_node.replace(exp.column(col_node.name, table="__THIS__"))

    for partition_cols in seen:
        table = state_table_name(partition_cols)
        if not partition_cols:
            on = f"{table}.{STATE_MARKER} = 0"
        else:
            on = " AND ".join(f"__THIS__.{c} = {table}.{c}" for c in partition_cols)
        select.join(exp.to_table(table), on=on, join_type="LEFT", copy=False)

    for table in extra_marker_tables:
        select.join(
            exp.to_table(table),
            on=f"{table}.{STATE_MARKER} = 0",
            join_type="LEFT",
            copy=False,
        )

    return select.sql()
