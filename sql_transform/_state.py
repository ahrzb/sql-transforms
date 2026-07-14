"""Extract learned state from DataFusion logical plans.

Parses DataFusion plan display text to find window aggregate columns in
projection expressions, then executes one DataFusion query per DISTINCT
(function, column) pair to compute its scalar value. The result is a
synthesized Pydantic model instance ("StateModel") keyed by `{fn}_{col}`
(no leading underscore -- see state_key), suitable for use as InferFn's
__STATE__ row table.
"""

from __future__ import annotations

import re

import datafusion
from pydantic import BaseModel

from sql_transform._schema import synthesize_state_model


def state_key(fn_name: str, col_name: str) -> str:
    """The __STATE__ field name for a given aggregate function + column,
    e.g. state_key("AVG", "age") == "avg_age". No leading underscore --
    Pydantic v2 would treat that as a private attribute."""
    return f"{fn_name.lower()}_{col_name.lower()}"


def extract_state(
    plan: datafusion.plan.LogicalPlan,
    ctx: datafusion.SessionContext,
    table_name: str,
) -> BaseModel:
    """Return a synthesized StateModel instance with one float field per
    distinct (fn, col) window aggregate referenced in the plan.

    Raises NotImplementedError if any window aggregate uses PARTITION BY --
    not yet supported by the Rust-backed pipeline.
    """
    display = plan.display_indent()

    pairs: dict[tuple[str, str], None] = {}
    for m in _WINDOW_AGG_RE.finditer(display):
        if m.group("partition"):
            raise NotImplementedError(
                "PARTITION BY window aggregates are not yet supported by "
                "the Rust-backed SQLTransform pipeline"
            )
        pairs[(m.group("fn").lower(), m.group("col").lower())] = None

    values: dict[str, float] = {}
    for fn_name, col_name in pairs:
        sql = f"SELECT {fn_name}({col_name}) FROM {table_name}"
        result = ctx.sql(sql).collect()
        value = result[0].column(0)[0].as_py()
        values[state_key(fn_name, col_name)] = float(value)

    state_model = synthesize_state_model(values)
    return state_model(**values)


_WINDOW_AGG_RE = re.compile(
    r"(?P<fn>\w+)"
    r"\((?:\w+)\.(?P<col>\w+)\)"
    r"(?P<partition>\s+PARTITION\s+BY\s+\[(?:\w+)\.\w+\])?"
    r"\s+ROWS\s+BETWEEN[^,\n]+"
)
