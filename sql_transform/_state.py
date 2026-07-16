"""Build typed per-partition state tables from SQLTransform's window aggregates.

Groups window aggregates by their PARTITION BY key-set and runs one DataFusion
GROUP BY query per group, producing a pyarrow state table (key columns + one
value column per distinct (fn, col)) whose value columns keep their natural
Arrow type -- no float coercion. The global OVER () group (empty key-set) gets a
one-row table plus a constant __state_marker__ column so the rewrite can LEFT
JOIN it uniformly. State tables are consumed as static tables by both engines
(InferFn static_tables and DataFusion registration) -- there is no Pydantic
state model anymore.
"""

from __future__ import annotations

import datafusion
import pyarrow as pa

from sql_transform._sql import WindowAgg

STATE_MARKER = "__state_marker__"


def state_key(fn_name: str, col_name: str) -> str:
    """The state value-column name for an aggregate function + column,
    e.g. state_key("AVG", "age") == "avg_age"."""
    return f"{fn_name.lower()}_{col_name.lower()}"


def state_table_name(partition_cols: tuple[str, ...]) -> str:
    """Deterministic state-table name for a partition-key-set. Empty key-set
    (the global OVER () state) -> "__STATE__"; otherwise
    "__STATE_BY_<cols joined by _>__"."""
    if not partition_cols:
        return "__STATE__"
    return "__STATE_BY_" + "_".join(partition_cols) + "__"


def build_state_tables(
    windows: list[WindowAgg],
    ctx: datafusion.SessionContext,
    table_name: str,
) -> dict[str, pa.Table]:
    """Return a dict of state-table-name -> pyarrow table, one per distinct
    PARTITION BY key-set present in `windows`. Value columns keep their real
    Arrow type. Raises NotImplementedError for ORDER BY window aggregates and
    ValueError for a case-collision between two aggregates in the same table."""
    # Group windows by partition-key-set, preserving discovery order.
    groups: dict[tuple[str, ...], list[WindowAgg]] = {}
    for w in windows:
        if w.has_order:
            raise NotImplementedError(
                "ORDER BY window aggregates are not yet supported by "
                "the Rust-backed SQLTransform pipeline"
            )
        groups.setdefault(w.partition_cols, []).append(w)

    tables: dict[str, pa.Table] = {}
    for partition_cols, members in groups.items():
        # Dedup by key within the group; detect state-key collisions.
        selected: dict[str, WindowAgg] = {}
        for w in members:
            existing = selected.get(w.key)
            if existing is not None and existing.arg.sql() != w.arg.sql():
                raise ValueError(
                    f"Ambiguous window aggregate: {w.fn}({w.arg.sql()}) normalizes "
                    f"to the same state key {w.key!r} as another aggregate in this "
                    "query"
                )
            selected[w.key] = w

        value_exprs = [f"{w.fn}({w.arg.sql()}) AS {key}" for key, w in selected.items()]

        if partition_cols:
            key_list = ", ".join(f'"{c}"' for c in partition_cols)
            sql = (
                f"SELECT {key_list}, {', '.join(value_exprs)} "
                f"FROM {table_name} GROUP BY {key_list}"
            )
            table = _collect(ctx, sql)
            # An unseen partition key yields NULL for every value column after
            # the LEFT JOIN, even for aggregates DataFusion types non-nullable
            # (COUNT is int64 NOT NULL). Widen value columns to nullable so
            # infer()'s output model accepts the NULL, matching transform().
            schema = pa.schema(
                [
                    f.with_nullable(True) if f.name not in partition_cols else f
                    for f in table.schema
                ]
            )
            table = table.cast(schema)
        else:
            sql = f"SELECT {', '.join(value_exprs)} FROM {table_name}"
            table = _collect(ctx, sql)
            table = table.append_column(STATE_MARKER, pa.array([0], type=pa.int64()))

        tables[state_table_name(partition_cols)] = table

    return tables


def _collect(ctx: datafusion.SessionContext, sql: str) -> pa.Table:
    df = ctx.sql(sql)
    return pa.Table.from_batches(df.collect(), schema=df.schema())
