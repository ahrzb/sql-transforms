"""Extract learned state from DataFusion logical plans.

Parses DataFusion plan display text to find window aggregate columns
in projection expressions, then executes separate queries to extract
constant values or partition-based lookup dicts.
"""

from __future__ import annotations

import re

import datafusion


def extract_state(
    plan: datafusion.plan.LogicalPlan,
    ctx: datafusion.SessionContext,
    table_name: str,
) -> dict:
    display = plan.display_indent()
    state: dict = {}

    for m in _WINDOW_AGG_RE.finditer(display):
        fn_name = m.group("fn").upper()
        col_name = m.group("col")
        partition_col = m.group("partition")
        segment = m.group(0)
        last_as = re.search(r"AS\s+(\w+)\s*$", segment)
        if not last_as:
            continue
        out_alias = last_as.group(1)

        if not partition_col:
            sql = f"SELECT {fn_name}({col_name}) FROM {table_name}"
            result = ctx.sql(sql).collect()
            value = result[0].column(0)[0].as_py()
            state[out_alias] = float(value)
        else:
            sql = (
                f"SELECT {partition_col}, {fn_name}({col_name}) "
                f"FROM {table_name} GROUP BY {partition_col}"
            )
            result = ctx.sql(sql).collect()
            keys: list = []
            vals: list = []
            for batch in result:
                keys.extend(batch.column(0).to_pylist())
                vals.extend(batch.column(1).to_pylist())
            state[out_alias] = {
                "lookup": dict(zip(keys, vals)),
                "partition_col": partition_col,
            }

    return state


_WINDOW_AGG_RE = re.compile(
    r"(?P<fn>\w+)"
    r"\((?:\w+)\.(?P<col>\w+)\)"
    r"(?:\s+PARTITION\s+BY\s+\[(?:\w+)\.(?P<partition>\w+)\])?"
    r"\s+ROWS\s+BETWEEN[^,\n]+"
)
