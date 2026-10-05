"""`error(msg)` as a CASE result: typed like a NULL arm, raising when taken.

DuckDB types `error()` as SQLNULL and never folds it, so inside a CASE it
adopts the CASE's type and raises with `Invalid Input Error: <msg>` exactly
on the rows that take its arm. That is the form served; anywhere else, and
with a message computed per row, it refuses by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest
from confit import SqlFunction
from confit import sql as S

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

ROWS = table(
    {"a": "int?", "x": "float?", "s": "str?"},
    [
        {"a": 0, "x": 0.5, "s": "p"},
        {"a": None, "x": None, "s": None},
        {"a": 2, "x": -1.0, "s": ""},
    ],
)
SAFE = ROWS.slice(0, 2)

DISPATCH = (
    "CASE WHEN a = 0 THEN x WHEN a = 2 THEN x * 2 WHEN a IS NULL THEN NULL "
    "ELSE error('no instance') END"
)


@pytest.mark.parametrize(
    "sql, rows",
    [
        (f"SELECT {DISPATCH} AS o FROM __THIS__", ROWS),
        (f"SELECT {DISPATCH} + 1 AS o FROM __THIS__", ROWS),
        (
            "SELECT CASE WHEN a = 0 THEN 1 ELSE error('it''s ' || 'bad') END AS o FROM __THIS__",
            SAFE,
        ),
        ("SELECT CASE WHEN a = 0 THEN 1 ELSE error(NULL) END AS o FROM __THIS__", ROWS),
        (
            "SELECT CASE WHEN a = 0 THEN s ELSE error(CAST(NULL AS VARCHAR)) END AS o FROM __THIS__",
            ROWS,
        ),
        (
            "SELECT CASE a WHEN 0 THEN 1.5 ELSE error('x') END AS o FROM __THIS__ WHERE a = 0",
            ROWS,
        ),
        (
            "SELECT CASE WHEN a > 5 THEN error('big') ELSE a END AS o FROM __THIS__",
            ROWS,
        ),
        (
            "SELECT a FROM __THIS__ WHERE CASE WHEN a = 0 THEN TRUE ELSE error('w') END",
            SAFE,
        ),
    ],
)
def test_an_error_arm_agrees_with_the_oracle(sql, rows):
    assert_parity(sql, rows)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT CASE WHEN a = 0 THEN 1 ELSE error('no instance') END AS o FROM __THIS__",
        "SELECT CASE WHEN TRUE THEN error('always') ELSE 1 END AS o FROM __THIS__",
    ],
)
def test_a_taken_error_arm_traps_on_both_engines(sql):
    assert_parity(sql, ROWS, trap="Invalid Input Error: (no instance|always)")


@pytest.mark.parametrize(
    "sql, why",
    [
        ("SELECT error('x') + 1 AS o FROM __THIS__", "outside a CASE result"),
        (
            "SELECT CASE WHEN a = 0 THEN 1 ELSE error('n' || s) END AS o FROM __THIS__",
            "per row",
        ),
        (
            "SELECT CASE WHEN a = 0 THEN error('x') END AS o FROM __THIS__",
            "every branch",
        ),
        (
            "SELECT CASE WHEN a = 0 THEN 1 ELSE error(5) END AS o FROM __THIS__",
            "VARCHAR",
        ),
    ],
)
def test_other_forms_refuse_by_name(sql, why):
    v = assert_parity(sql, ROWS, expect="REFUSED")
    assert why in v.detail


def test_an_instance_dispatch_in_a_sql_function():
    # The native catalog's shape: a NULL id answers NULL, a known id its
    # instance's value, an unknown id raises.
    def body(iid, v):
        return (
            S.case(iid == S.lit(0), v * S.lit(2.0))
            .when(iid.isnull(), S.lit(None, pa.float64()))
            .otherwise(S.fn("error", S.lit("instance id not fitted")))
        )

    f = SqlFunction(
        "tf", pa.schema([("iid", pa.int64()), ("v", pa.float64())]), pa.float64(), body
    )
    sql = "SELECT tf(a, x) AS o FROM __THIS__"
    assert_parity(sql, SAFE, udfs=[f])
    assert_parity(sql, ROWS, udfs=[f], trap="instance id not fitted")
