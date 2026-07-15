"""Extract learned state from SQLTransform's window aggregates.

Runs one DataFusion query per DISTINCT (function, column) pair found by
_sql.py's find_window_aggregates(), and synthesizes a typed Pydantic
model instance ("StateModel") keyed by `{fn}_{col}` (no leading
underscore -- see state_key), suitable for use as InferFn's __STATE__ row
table. DataFusion's only role here is executing those small per-aggregate
queries -- it never parses or plans SQLTransform's original SQL.
"""

from __future__ import annotations

import datafusion
from pydantic import BaseModel

from sql_transform._schema import synthesize_state_model
from sql_transform._sql import WindowAgg


def state_key(fn_name: str, col_name: str) -> str:
    """The __STATE__ field name for a given aggregate function + column,
    e.g. state_key("AVG", "age") == "avg_age". No leading underscore --
    Pydantic v2 would treat that as a private attribute."""
    return f"{fn_name.lower()}_{col_name.lower()}"


def extract_state(
    windows: list[WindowAgg],
    ctx: datafusion.SessionContext,
    table_name: str,
) -> BaseModel:
    """Return a synthesized StateModel instance with one float field per
    distinct (fn, col) window aggregate in `windows`.

    Raises NotImplementedError if any window aggregate uses PARTITION BY
    or ORDER BY -- not yet supported by the Rust-backed pipeline.
    """
    pairs: dict[tuple[str, str], None] = {}
    for w in windows:
        if w.has_partition:
            raise NotImplementedError(
                "PARTITION BY window aggregates are not yet supported by "
                "the Rust-backed SQLTransform pipeline"
            )
        if w.has_order:
            raise NotImplementedError(
                "ORDER BY window aggregates are not yet supported by "
                "the Rust-backed SQLTransform pipeline"
            )
        # Preserve the column's real case here -- lower-casing it would
        # break the query below against a mixed-case schema, and would
        # collide two distinct case-differing columns onto the same
        # dedup key (state_key() below normalizes the STATE FIELD name
        # to lowercase, which is a separate, intentional choice).
        pairs[(w.fn, w.col)] = None

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
