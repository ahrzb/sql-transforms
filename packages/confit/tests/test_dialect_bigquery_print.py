"""The BigQuery print target through the Python boundary.

Spelling-level coverage only: the printer's semantics follow BigQuery's
documented GoogleSQL and the design's landing-zone table, and the live
remote gate is the design's phase 4 — still owed. What this test pins is
that the target is reachable from Python, forces the documented spellings
(typed decimal literals, SAFE_CAST, DIV/MOD), and refuses the design's
refusal rows by name.
"""

from __future__ import annotations

import duckdb
import pytest
from confit import _engine


@pytest.fixture
def catalog():
    con = duckdb.connect()
    con.execute("CREATE TABLE t(a INTEGER, b VARCHAR, c DOUBLE, big BIGINT)")
    cols = [
        (name, dtype, nullable == "YES")
        for name, dtype, nullable, *_ in con.execute('DESCRIBE "t"').fetchall()
    ]
    return [("t", cols)]


def to_bq(sql: str, catalog) -> str:
    plan_text = _engine.dialect_parse(sql, catalog)
    return _engine.dialect_print(plan_text, "bigquery", catalog)


def test_forced_spellings(catalog):
    printed = to_bq(
        "SELECT big // 2 AS q, 1.5 / c AS d, TRY_CAST(a AS DOUBLE) AS w FROM t "
        "WHERE b IS NOT DISTINCT FROM 'o''k'",
        catalog,
    )
    # Forced spellings after the adversarial review: DIV inside the
    # zero-divisor guard (DuckDB gives NULL where BigQuery DIV errors),
    # / as IEEE_DIVIDE (DuckDB's zero divisors are IEEE inf/NaN).
    guarded_div = "(CASE WHEN 2 = 0 THEN CAST(NULL AS INT64) ELSE DIV(`big`, 2) END)"
    assert f"{guarded_div} AS `q`" in printed
    assert "IEEE_DIVIDE(CAST(NUMERIC '1.5' AS FLOAT64), `c`) AS `d`" in printed
    assert "SAFE_CAST(`a` AS FLOAT64) AS `w`" in printed
    assert "(`b` IS NOT DISTINCT FROM 'o\\'k')" in printed


def test_named_refusals(catalog):
    # Narrow-int arithmetic computes at i32 in DuckDB (trap at 2^31) —
    # unforceable on INT64-only BigQuery until guard expressions land.
    with pytest.raises(ValueError, match="unsupported: bigquery: .*INT64"):
        to_bq("SELECT a + 1 AS x FROM t", catalog)
    # INT64 arithmetic shares the overflow-error class: allowed.
    assert "(`big` + 1)" in to_bq("SELECT big + 1 AS x FROM t", catalog)
