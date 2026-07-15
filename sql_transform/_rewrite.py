"""Rewrite SQLTransform's SQL into SQL runnable by the Rust InferFn.

Given the Select tree and WindowAgg list from _sql.py, replaces every
window-aggregate reference with a __STATE__ column reference, qualifies
every plain column as __THIS__.col, and appends __STATE__ as a cross-
joined FROM entry (__STATE__ is always exactly one row, so a cross join
just repeats it against every __THIS__ row).
"""

from __future__ import annotations

from sqlglot import exp

from sql_transform._sql import WindowAgg
from sql_transform._state import state_key


def rewrite_sql(select: exp.Select, windows: list[WindowAgg]) -> str:
    """Return SQL text equivalent to `select`'s projection, with every
    window-aggregate reference replaced by a __STATE__ column reference.

    Mutates `select` in place via node.replace() -- callers should not
    reuse `select` afterwards.
    """
    window_key = {id(w.node): state_key(w.fn, w.col) for w in windows}

    for e in select.expressions:
        out_name = e.alias_or_name
        if not out_name:
            raise ValueError(
                "Expression in SELECT list needs an alias (AS name): "
                f"{e.sql()!r}"
            )

        for win_node in list(e.find_all(exp.Window)):
            win_node.replace(
                exp.column(window_key[id(win_node)], table="__STATE__")
            )

        for col_node in list(e.find_all(exp.Column)):
            if col_node.table == "__STATE__":
                continue  # already rewritten by the pass above
            if col_node.table and col_node.table != "__THIS__":
                raise ValueError(
                    f"Column qualifier {col_node.table!r} does not refer "
                    "to __THIS__"
                )
            col_node.replace(exp.column(col_node.name, table="__THIS__"))

    select.join("__STATE__", join_type="CROSS", copy=False)
    return select.sql()
