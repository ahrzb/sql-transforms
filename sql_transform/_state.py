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
        spec = m.group("spec")
        if "PARTITION BY" in spec:
            raise NotImplementedError(
                "PARTITION BY window aggregates are not yet supported by "
                "the Rust-backed SQLTransform pipeline"
            )
        if "ORDER BY" in spec:
            raise NotImplementedError(
                "ORDER BY window aggregates are not yet supported by "
                "the Rust-backed SQLTransform pipeline"
            )
        # Preserve the column's real case here -- lower-casing it would
        # break the query below against a mixed-case schema, and would
        # collide two distinct case-differing columns onto the same
        # dedup key (state_key() below normalizes the STATE FIELD name
        # to lowercase, which is a separate, intentional choice).
        pairs[(m.group("fn"), m.group("col"))] = None

    values: dict[str, float] = {}
    for fn_name, col_name in pairs:
        # Quote the column name so DataFusion resolves it against the
        # schema's real (possibly mixed-case) field name rather than
        # case-folding an unquoted identifier to lowercase.
        sql = f'SELECT {fn_name}("{col_name}") FROM {table_name}'
        result = ctx.sql(sql).collect()
        value = result[0].column(0)[0].as_py()
        key = state_key(fn_name, col_name)
        if key in values:
            raise ValueError(
                f"Ambiguous window aggregate: {fn_name}({col_name}) "
                f"normalizes to the same state key {key!r} as another "
                "aggregate in this query -- column names that differ only "
                "by case aren't distinguished"
            )
        values[key] = float(value)

    state_model = synthesize_state_model(values)
    return state_model(**values)


_WINDOW_AGG_RE = re.compile(
    r"(?P<fn>\w+)"
    r"\((?:\w+)\.(?P<col>\w+)\)"
    r"(?P<spec>.*?)"
    r"(?:ROWS|RANGE|GROUPS)\s+BETWEEN[^,\n]+"
)
