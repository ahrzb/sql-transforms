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

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn


# ---------------------------------------------------------------------------
# TASK-121: a bare struct path whose HEAD name also binds in a join scope is
# ambiguous on DuckDB and binds here. 16 of the 28 findings in the 2026-08-17
# campaign, and the largest single class the fuzzer sees.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="TASK-121: `ON (c0.f0 = s0.c0)` where __THIS__.c0 is a struct and "
    "s0.c0 is a column -- DuckDB refuses as ambiguous before it looks at .f0; "
    "we resolve to the struct and answer. The BARE-column case already refuses; "
    "the struct-path route does not consult the join scopes.",
)
def test_an_ambiguous_struct_path_refuses():
    row = pa.schema(
        [
            pa.field("c0", pa.struct([("f0", pa.int64())])),
            pa.field("v", pa.int64(), nullable=False),
        ]
    )
    static = pa.table({"c0": pa.array([1], pa.int64()), "w": pa.array([9], pa.int64())})
    sql = "SELECT v AS o FROM __THIS__ LEFT JOIN s0 ON (c0.f0 = s0.c0)"

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (c0 STRUCT(f0 BIGINT), v BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES ({'f0': 1}, 2)")
    con.register("sa", static)
    con.execute("CREATE TABLE s0 AS SELECT * FROM sa")
    with pytest.raises(duckdb.Error, match="[Aa]mbiguous"):
        con.execute(sql).fetchall()  # oracle refuses; if this stops, remeasure

    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s0": static})


# ---------------------------------------------------------------------------
# TASK-130: a join star does not dedupe colliding output names. Found by the
# campaign once TASK-124's grammar shift re-rolled seed 380; pre-existing.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="TASK-130: DuckDB renames the second `c0` in a join star to `c0_1`; "
    "we emit the duplicate name, which a row DICT then silently collapses.",
)
def test_a_join_star_dedupes_colliding_names():
    row = pa.schema(
        [pa.field("c0", pa.int64(), nullable=False), pa.field("c2", pa.int64())]
    )
    static = pa.table(
        {"c0": pa.array([1], pa.int64()), "c1": pa.array([9], pa.int64())}
    )
    sql = "SELECT * FROM __THIS__ LEFT JOIN s0 ON (c2 = s0.c0)"

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (c0 BIGINT, c2 BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (5, 1)")
    con.register("sa", static)
    con.execute("CREATE TABLE s0 AS SELECT * FROM sa")
    res = con.execute(sql)
    names = [d[0] for d in res.description]
    assert names == ["c0", "c2", "c0_1", "c1"]  # oracle dedupes; remeasure if not

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s0": static})
    assert fn.output_schema.names == names
