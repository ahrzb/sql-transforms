"""DataFusion batch execution for a fitted SQLTransform.

Registers __THIS__ plus every fit-time state table, then runs the rewritten SQL
(LEFT JOINs to those state tables). DataFusion yields NULL on an unseen partition
natively -- matching the Rust engine's LEFT-lookup-join. This is the vectorized
counterpart to the row-at-a-time InferFn path.
"""

from __future__ import annotations

import datafusion
import pyarrow as pa
import sqlglot

from sql_transform._transformer_udf import _transformer_udf

_ROW_ID = "__row_id__"


def run_batch(
    rewritten_sql: str,
    table: pa.Table,
    state_tables: dict[str, pa.Table],
    transformers: dict[str, tuple[object, pa.Schema, pa.Schema]] | None = None,
) -> pa.Table:
    """Execute `rewritten_sql` against `table` (as __THIS__) and every state
    table (registered by name) via DataFusion, returning a pyarrow Table.

    DataFusion's LEFT JOIN physical plan is free to build its hash table from
    either side, so the output row order isn't guaranteed to match __THIS__'s
    input order once a partitioned state table is joined in. We tag each input
    row with a row id, ask DataFusion to ORDER BY it (a real sort, unlike join
    order, so this is deterministic), and drop the id column before returning
    -- restoring the strict 1-to-1, input-order-preserving contract."""
    if _ROW_ID in table.column_names:
        raise ValueError(
            f"__THIS__ must not contain a reserved column named {_ROW_ID!r}"
        )
    numbered_table = table.append_column(
        _ROW_ID, pa.array(range(table.num_rows), type=pa.int64())
    )

    tree = sqlglot.parse_one(rewritten_sql)
    tree.select(f"__THIS__.{_ROW_ID} AS {_ROW_ID}", copy=False)
    tree = tree.order_by(_ROW_ID, copy=False)

    ctx = datafusion.SessionContext()
    ctx.from_arrow(numbered_table, name="__THIS__")
    for name, state_table in state_tables.items():
        ctx.from_arrow(state_table, name=name)
    for name, (obj, in_schema, out_schema) in (transformers or {}).items():
        ctx.register_udf(_transformer_udf(obj, in_schema, out_schema, name))
    df = ctx.sql(tree.sql())
    # collect() returns [] for a zero-row result, so pass the schema explicitly.
    out = pa.Table.from_batches(df.collect(), schema=df.schema())
    return out.drop([_ROW_ID])
