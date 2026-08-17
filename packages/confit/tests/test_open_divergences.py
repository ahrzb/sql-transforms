"""Divergences we intend to CLOSE — one xfail-strict pin each, ticket named.

The split from `test_known_divergences.py` is by INTENT, not by severity:

    test_known_divergences.py   behaviour we have decided to KEEP. Every
                                entry states the ground for keeping it, and
                                its tests PASS - they are regression pins on
                                a settled answer.

    this file                   behaviour we have decided to CHANGE. Every
                                entry is xfail(strict=True) and names the
                                task that closes it. When the fix lands the
                                pin flips loudly, and the entry is deleted
                                rather than edited.

Why the separation is worth the file: mixing the two makes "is this on
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
# TASK-115: a projected static column takes the JOIN KEY's width instead of
# its own. Found by the widened fuzzer (seed 1379) while verifying TASK-96.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="TASK-115: projecting the static side of a join KEY reconstructs "
    "it from the dynamic side, so it takes the ROW column's width instead of "
    "the static column's own. int8 row key vs int64 static key emits int8; "
    "DuckDB emits int64. Non-key static columns are unaffected.",
)
def test_projected_static_key_keeps_its_own_width():
    row = pa.schema([pa.field("k", pa.int8(), nullable=False)])
    static = pa.table({"c0": pa.array([5], pa.int64()), "v": pa.array([7], pa.int64())})
    sql = "SELECT s.c0 AS o FROM __THIS__ LEFT JOIN s ON k = s.c0"

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k TINYINT)")
    con.execute("INSERT INTO __THIS__ VALUES (5)")
    con.register("sa", static)
    con.execute("CREATE TABLE s AS SELECT * FROM sa")
    want = con.execute(sql).to_arrow_table()

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s": static})
    got = fn.infer_arrow(pa.Table.from_pylist([{"k": 5}], schema=row))
    assert want.schema.field("o").type == pa.int64(), "oracle moved — remeasure"
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"


# ---------------------------------------------------------------------------
# TASK-116: a struct column is lanes in a row table and unserved in a static
# one. Reported 2026-08-15; #157 made the refusal name the type, this pin is
# for actually serving it.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="TASK-116: struct columns serve in a ROW table (TASK-56 flattens "
    "them to lanes) and are unserved in a STATIC one, so the same column "
    "binds or refuses depending only on which table it sits in.",
)
def test_struct_static_column_serves_its_lanes():
    row = pa.schema([pa.field("k", pa.int64(), nullable=False)])
    static = pa.table(
        {
            "id": pa.array([5], pa.int64()),
            "w": pa.array([{"mean": 2.0}], pa.struct([("mean", pa.float64())])),
        }
    )
    sql = "SELECT s.w.mean AS o FROM __THIS__ JOIN s ON k = s.id"

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (5)")
    con.register("sa", static)
    con.execute("CREATE TABLE s AS SELECT * FROM sa")
    want = con.execute(sql).to_arrow_table().to_pylist()

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s": static})
    assert fn.infer_rows([{"k": 5}]) == want


@pytest.mark.xfail(
    strict=True,
    reason="TASK-117: `trapping_expr BETWEEN lit AND NULL` is statically NULL, "
    "so DuckDB never evaluates the subject and the row simply filters. We "
    "evaluate the subject first and trap. TASK-85 folds a strict op whose "
    "SIBLING is NULL; here the trap is the thing being compared.",
)
def test_a_null_folded_predicate_elides_its_subject_too():
    row = pa.schema([pa.field("s", pa.string(), nullable=False)])
    rows = [{"s": "abc"}]  # not castable to DOUBLE — the trap
    sql = (
        "SELECT 1 AS o FROM __THIS__ WHERE CAST(s AS DOUBLE) BETWEEN 61.591e0 AND NULL"
    )
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (s VARCHAR)")
    con.execute("INSERT INTO __THIS__ VALUES ('abc')")
    want = con.execute(sql).fetchall()
    assert want == [], "oracle moved — remeasure"

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={})
    assert [tuple(r.values()) for r in fn.infer_rows(rows)] == want


# ---------------------------------------------------------------------------
# TASK-118: an INT32 overflow consumed by a WIDENING CAST serves a wrong value.
# TASK-84's block names `CAST(k AS INTEGER) * 2` as its residual; that case is
# caught now. This one is not, and it is worse than a missed trap - we return a
# number DuckDB never produces, with no refusal anywhere.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="TASK-118: (i + 1) over INT32_MAX overflows INT32 on DuckDB and "
    "traps. Widening the result to BIGINT hides it here: we compute in the i64 "
    "lane and serve 2147483648, a value DuckDB never returns.",
)
def test_int32_overflow_under_a_widening_cast_does_not_serve_a_wrong_value():
    row = pa.schema([pa.field("i", pa.int32(), nullable=False)])
    rows = [{"i": 2147483647}]
    sql = "SELECT CAST((i + 1) AS BIGINT) AS o FROM __THIS__"

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (i INTEGER)")
    con.execute("INSERT INTO __THIS__ VALUES (2147483647)")
    with pytest.raises(duckdb.Error):
        con.execute(sql).fetchall()  # oracle traps; if this stops, remeasure

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={})
    with pytest.raises(ValueError, match="int32|INT32|[Oo]verflow"):
        fn.infer_rows(rows)
