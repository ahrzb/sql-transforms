"""List literals, `[..]` and `list_value(..)`, read by a constant index or
projected whole; and list-valued SQL functions built on them.

DuckDB builds every element (a sibling's trap fires on an element read),
indexes from 1, counts a negative index from the end, and answers NULL at 0
and past either end (measured on 1.5.5, optimizer off).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest
from confit import FunctionError, SqlFunction
from confit import sql as S

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

ROWS = table(
    {"a": "int?", "x": "float?", "s": "str?"},
    [
        {"a": 2, "x": 1.5, "s": "abc"},
        {"a": None, "x": None, "s": None},
        {"a": 0, "x": -1.0, "s": ""},
    ],
)
SAFE = ROWS.slice(1, 2)
BIG = "a * 9223372036854775807"
B7 = "CAST(7 AS BIGINT)"


def q(expr: str) -> str:
    return f"SELECT {expr} AS o FROM __THIS__"


@pytest.mark.parametrize("i", [-3, -2, -1, 0, 1, 2, 3])
def test_an_index_reads_like_duckdb(i):
    assert_parity(q(f"[a, {B7}][{i}]"), ROWS)
    assert_parity(q(f"list_extract(list_value(x, x * 2), {i})"), ROWS)


@pytest.mark.parametrize(
    "expr",
    [
        f"[a, {BIG}][1]",
        f"[{BIG}, a][2]",
        f"list_value(a, {BIG})[5]",
        f"array_extract([a, {BIG}], 1)",
    ],
)
def test_a_sibling_element_still_traps(expr):
    assert_parity(q(expr), ROWS, trap="Overflow")
    assert_parity(q(expr), SAFE)


@pytest.mark.parametrize(
    "expr",
    [
        "[CAST(a AS DOUBLE), x]",
        "list_value(s, s || 'z', NULL)",
        "CASE WHEN a IS NULL THEN NULL ELSE [CAST(a AS DOUBLE), x] END",
        "(CASE WHEN a > 0 THEN [x, x] END)[1]",
        "(CASE WHEN a > 0 THEN s ELSE 'zz' END)[2]",
    ],
)
def test_a_list_agrees_with_the_oracle(expr):
    assert_parity(q(expr), ROWS)


LST = SqlFunction(
    "lst",
    pa.schema([("i", pa.int64()), ("v", pa.float64())]),
    pa.list_(pa.float64(), 3),
    lambda i, v: [v, v * S.lit(2.0), -v],
    null_when=lambda i, v: i.isnull(),
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT lst(a, x) AS o FROM __THIS__",
        "SELECT lst(a, x)[2] AS o, lst(a, x)[-1] AS p, lst(a, x)[4] AS r FROM __THIS__",
        "SELECT list_extract(lst(0, x), 3) AS o FROM __THIS__",
    ],
)
def test_a_list_sql_function_agrees_with_the_oracle(sql):
    assert_parity(sql, ROWS, udfs=[LST])


@pytest.mark.parametrize(
    "expr, why",
    [
        ("[a, x]", "different types"),
        ("[a, 7][2]", "different types"),
        ("[x]", "one-element list"),
        ("[x, x][a]", "not an integer constant"),
        ("[NULL, NULL][1]", "only NULLs"),
    ],
)
def test_other_forms_refuse_by_name(expr, why):
    v = assert_parity(q(expr), ROWS, expect="REFUSED")
    assert why in v.detail


def test_a_list_return_declares_its_body_width():
    with pytest.raises(FunctionError, match="list of 2"):
        SqlFunction(
            "f",
            pa.schema([("v", pa.float64())]),
            pa.list_(pa.float64(), 2),
            lambda v: [v],
        )
