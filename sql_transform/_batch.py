"""DataFusion batch execution for a fitted SQLTransform.

Runs the rewritten SQL (`__THIS__ CROSS JOIN __STATE__`) over a batch table,
using the frozen fit-time state as a one-row `__STATE__` table. This is the
vectorized counterpart to the row-at-a-time Rust `InferFn` path -- same
rewritten SQL, same frozen state, so both produce identical values on the
normal numeric path.
"""

from __future__ import annotations

import datafusion
import pyarrow as pa
from pydantic import BaseModel


def run_batch(
    rewritten_sql: str,
    table: pa.Table,
    state: BaseModel,
) -> pa.Table:
    """Execute `rewritten_sql` against `table` (registered as __THIS__) and the
    frozen `state` (registered as a one-row __STATE__ table) via DataFusion,
    returning the result as a pyarrow Table."""
    ctx = datafusion.SessionContext()
    ctx.from_arrow(table, name="__THIS__")
    ctx.from_arrow(_state_to_table(state), name="__STATE__")
    df = ctx.sql(rewritten_sql)
    # collect() returns [] for a zero-row result, and pa.Table.from_batches([])
    # raises -- so pass the DataFrame's schema explicitly to preserve it.
    return pa.Table.from_batches(df.collect(), schema=df.schema())


def _state_to_table(state: BaseModel) -> pa.Table:
    """Build the one-row __STATE__ table from a frozen state model.

    A zero-field state (a query with no window aggregates) would produce a
    zero-column Arrow table, which cannot hold one row; emit a single
    placeholder column instead. The rewritten SQL never selects it, so it does
    not appear in the output."""
    data = {key: [value] for key, value in state.model_dump().items()}
    if not data:
        data = {"__state_marker__": [0]}
    return pa.table(data)
