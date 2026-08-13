"""The integer-widths feature (m-8 phase 2, TASK-79).

DuckDB types in a width lattice (TINYINT..HUGEINT); this engine typed every
integer BIGINT, so infer_arrow's schema diverged wherever DuckDB says
INTEGER (literals, ascii, CASE over literals, ::INTEGER casts). Phase 2
types the widths for real in the frontend, erased to the i64 lane at
compute; the width is observable exactly where DuckDB's is — the trap
threshold (phase 3) and the Arrow schema (here).

Every schema expectation below is the live oracle, never a hardcoded width:
the assert is `ours == DuckDB's`, so a wrong row here is impossible.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from pydantic import create_model

In = create_model("In", k=(int, ...), s=(str, ...))
ROWS = [{"k": 0, "s": "ab"}, {"k": 2, "s": "Z"}, {"k": 30000, "s": "!"}]

# The measured catalogue (spec 2026-08-11 + probe 2026-08-13). Mix of rows
# that must NARROW (int32/int16/int8) and controls that must stay int64 or
# double — the oracle decides which is which.
CATALOGUE = [
    "SELECT 1 AS o FROM __THIS__",
    "SELECT 2147483647 AS o FROM __THIS__",
    "SELECT 2147483648 AS o FROM __THIS__",
    "SELECT -2147483647 AS o FROM __THIS__",
    "SELECT -2147483648 AS o FROM __THIS__",  # BIGINT: parsed -(2147483648)
    "SELECT CASE WHEN k > 1 THEN 1 ELSE 0 END AS o FROM __THIS__",
    "SELECT CASE WHEN k > 1 THEN 1 ELSE k END AS o FROM __THIS__",
    "SELECT 1 + 2 AS o FROM __THIS__",
    "SELECT 2000000000 + 2000000 AS o FROM __THIS__",
    "SELECT 7 // 2 AS o FROM __THIS__",
    "SELECT 7 % 2 AS o FROM __THIS__",
    "SELECT 7 / 2 AS o FROM __THIS__",
    "SELECT 1 & 2 AS o FROM __THIS__",
    "SELECT 1 | 2 AS o FROM __THIS__",
    "SELECT 1 << 2 AS o FROM __THIS__",
    "SELECT xor(1, 2) AS o FROM __THIS__",
    "SELECT k + 1 AS o FROM __THIS__",
    "SELECT k & 1 AS o FROM __THIS__",
    "SELECT -(3) AS o FROM __THIS__",
    "SELECT ascii(s) AS o FROM __THIS__",
    "SELECT unicode(s) AS o FROM __THIS__",
    "SELECT ord(s) AS o FROM __THIS__",
    "SELECT length(s) AS o FROM __THIS__",
    "SELECT bit_length(s) AS o FROM __THIS__",
    "SELECT strpos(s, 'a') AS o FROM __THIS__",
    "SELECT levenshtein(s, 'ab') AS o FROM __THIS__",
    "SELECT abs(-5) AS o FROM __THIS__",
    "SELECT abs(k) AS o FROM __THIS__",
    "SELECT round(1) AS o FROM __THIS__",
    "SELECT trunc(1) AS o FROM __THIS__",
    "SELECT round(1, 0) AS o FROM __THIS__",
    "SELECT greatest(1, 2) AS o FROM __THIS__",
    "SELECT least(1, 2) AS o FROM __THIS__",
    "SELECT greatest(1, k) AS o FROM __THIS__",
    "SELECT coalesce(1, k) AS o FROM __THIS__",
    "SELECT coalesce(1, 2) AS o FROM __THIS__",
    "SELECT nullif(1, k) AS o FROM __THIS__",
    "SELECT nullif(k, 1) AS o FROM __THIS__",
    "SELECT nullif(2, 3) AS o FROM __THIS__",
    "SELECT NULL AS o FROM __THIS__",
    "SELECT nullif(NULL, 84.754e0) AS o FROM __THIS__",
    "SELECT nullif(NULL, k) AS o FROM __THIS__",
    "SELECT nullif(NULL, NULL) AS o FROM __THIS__",
    "SELECT CAST(k AS INTEGER) AS o FROM __THIS__",
    "SELECT CAST(k AS SMALLINT) AS o FROM __THIS__",
    "SELECT CAST(1 AS TINYINT) AS o FROM __THIS__",
    "SELECT CAST(k AS BIGINT) AS o FROM __THIS__",
    "SELECT TRY_CAST(k AS INTEGER) AS o FROM __THIS__",
    "SELECT CASE WHEN k > 1 THEN ascii(s) ELSE 0 END AS o FROM __THIS__",
    "SELECT 1 BETWEEN 0 AND k AS o FROM __THIS__",
    "SELECT CAST(k AS INTEGER) % 24 AS o FROM __THIS__",
]


def _duck(sql: str) -> pa.Table:
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT, s VARCHAR)")
    for r in ROWS:
        con.execute("INSERT INTO __THIS__ VALUES (?, ?)", [r["k"], r["s"]])
    return con.execute(sql).to_arrow_table()


def _ours(sql: str) -> pa.Table:
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": In}, static_tables={}, output="dict"
    )
    return fn.infer_arrow(pa.Table.from_pylist(ROWS))


@pytest.mark.parametrize("sql", CATALOGUE)
def test_output_width_matches_duckdb(sql):
    got, want = _ours(sql), _duck(sql)
    assert got.to_pylist() == want.to_pylist(), sql
    assert got.schema == want.schema, f"{sql}: {got.schema} != {want.schema}"


def test_try_cast_to_integer_nulls_out_of_range():
    """TRY_CAST out of the target's range is NULL on DuckDB — not a trap, so
    it is phase-2 value semantics, not phase-3 trap work."""
    sql = "SELECT TRY_CAST(k AS INTEGER) AS o FROM __THIS__"
    big = [{"k": 9007199254740993, "s": "x"}, {"k": 5, "s": "y"}]
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": In}, static_tables={}, output="dict"
    )
    got = fn.infer({"__THIS__": [In(**r) for r in big]})
    assert [r["o"] for r in got] == [None, 5]


def test_out_of_range_dynamic_int32_refuses_at_emit_not_wraps():
    """CAST(k AS INTEGER) on an out-of-range k TRAPS on DuckDB. Our dynamic
    trap is phase 3; until then the int32 EMIT must refuse by name rather
    than wrap — every input this refuses is an input DuckDB errors on too."""
    sql = "SELECT CAST(k AS INTEGER) AS o FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": In}, static_tables={}, output="dict"
    )
    with pytest.raises(Exception, match="int32|INT32|INTEGER"):
        fn.infer_arrow(pa.Table.from_pylist([{"k": 9007199254740993, "s": "x"}]))
