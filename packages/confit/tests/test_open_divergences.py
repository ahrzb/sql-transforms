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
# TASK-131: a bare-NULL arm floors a CASE's width at INTEGER on DuckDB.
# Surfaced when TASK-130's oracle fix stopped the schema-name mismatch from
# returning early (seed 12745).
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="TASK-131: a bare NULL arm contributes SQLNULL/INTEGER to DuckDB's "
    "CASE unification, flooring the result at int32; we adopt the NULL into "
    "the value arms' width (int16).",
)
def test_a_bare_null_arm_floors_the_case_width():
    row = pa.schema([pa.field("c0", pa.int16(), nullable=False)])
    sql = (
        "SELECT (CASE WHEN TRUE THEN NULL WHEN TRUE THEN c0 ELSE -22 END) AS o "
        "FROM __THIS__"
    )
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (c0 SMALLINT)")
    con.execute("INSERT INTO __THIS__ VALUES (3)")
    want = con.execute(sql).to_arrow_table()
    assert want.schema.field("o").type == pa.int32(), "oracle moved — remeasure"

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={})
    assert fn.output_schema.field("o").type == pa.int32()
