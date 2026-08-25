"""Divergences we intend to CLOSE — one xfail-strict pin each, ticket named.

It emptied on 2026-08-17 when TASK-115, 116, 117, 118 and 119 all closed, and
refilled the same day from the campaign that followed the oracle change --
which is the intended rhythm, not churn. Adding a pin here is how a new
divergence gets recorded; emptying it again is what closing one looks like.

The split from `known_divergences/` is by INTENT, not by severity:

    known_divergences/   behaviour we have decided to KEEP. Every entry
                         states the ground for keeping it, and its tests
                         PASS — they are regression pins on a settled
                         answer.

    this file            behaviour we have decided to CHANGE. Every entry is
                         xfail(strict=True) and names the task that closes
                         it. When the fix lands the pin flips loudly, and
                         the entry is deleted rather than edited.

Why the separation is worth a second file: mixing the two makes "is this on
purpose?" unanswerable at a glance, and a reader who assumes the wrong one
either implements something we chose not to have, or leaves a real bug
sitting under a paragraph explaining why it is fine. The census on
2026-08-16 found both mistakes already present.

strict=True is the load-bearing part. A pin that silently starts passing is
worse than no pin: it certifies work nobody did.
"""

import datetime

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn

# TASK-133: NATURAL and USING joins will key on non-scalar shared columns
# like DuckDB does, rather than refusing them.
#
# DuckDB's NATURAL intersects column NAME SETS with no type inspection
# (bind_joinref.cpp:185-208), so a shared STRUCT -- or any shared column
# whose type this engine does not serve -- is a join key there. Our NATURAL
# arm iterates the scalar lane list, where neither kind appears, so both
# silently drop OUT of the key set and we emit rows DuckDB does not. That is
# a wrong ANSWER, not a refusal. Measured 2026-08-25; it predates TASK-132
# (a struct head used to be an `opaque` entry and the same loop skipped it).
_NATURAL133 = "SELECT z AS o FROM __THIS__ NATURAL JOIN s"
_T0 = datetime.datetime(2020, 1, 1)
_T1 = datetime.datetime(2021, 6, 30)


@pytest.mark.xfail(
    strict=True,
    reason="TASK-133: NATURAL JOIN drops unkeyable shared columns from the "
    "key set; the fix keys on them like DuckDB, it does not refuse them",
)
@pytest.mark.parametrize(
    ("row_ddl", "row_schema", "row_values", "row", "static"),
    [
        # a shared STRUCT column, values UNEQUAL: DuckDB keys on it and
        # returns nothing; we key on `id` alone and return the row
        (
            "id BIGINT, w STRUCT(mean DOUBLE)",
            pa.schema(
                [
                    pa.field("id", pa.int64(), nullable=False),
                    pa.field("w", pa.struct([("mean", pa.float64())])),
                ]
            ),
            "(5, {'mean': 1.0})",
            {"id": 5, "w": {"mean": 1.0}},
            pa.table(
                {
                    "id": pa.array([5], pa.int64()),
                    "w": pa.array([{"mean": 2.0}], pa.struct([("mean", pa.float64())])),
                    "z": pa.array([7], pa.int64()),
                }
            ),
        ),
        # not struct-specific: any shared column we cannot serve as a scalar
        (
            "id BIGINT, t TIMESTAMP",
            pa.schema(
                [
                    pa.field("id", pa.int64(), nullable=False),
                    pa.field("t", pa.timestamp("us")),
                ]
            ),
            "(5, TIMESTAMP '2020-01-01 00:00:00')",
            {"id": 5, "t": _T0},
            pa.table(
                {
                    "id": pa.array([5], pa.int64()),
                    "t": pa.array([_T1], pa.timestamp("us")),
                    "z": pa.array([7], pa.int64()),
                }
            ),
        ),
    ],
)
def test_a_natural_join_keys_on_every_shared_column(
    row_ddl, row_schema, row_values, row, static
):
    con = duckdb.connect()
    con.execute(f"CREATE TABLE __THIS__ ({row_ddl})")
    con.execute(f"INSERT INTO __THIS__ VALUES {row_values}")
    con.register("sa", static)
    con.execute("CREATE TABLE s AS SELECT * FROM sa")
    want = [{"o": r[0]} for r in con.execute(_NATURAL133).fetchall()]
    assert want == [], "oracle moved -- it no longer keys on the shared column"
    got = DuckDBInferFn(
        _NATURAL133, row_tables={"__THIS__": row_schema}, static_tables={"s": static}
    ).infer_rows([row])
    assert got == want
