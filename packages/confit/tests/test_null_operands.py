"""A NULL operand makes an integer operator NULL; it never makes it trap.

DuckDB answers NULL for `x op NULL` whatever `x` is, even where the same
operator with a value in NULL's place would overflow or reject its operand
(`0 - i64::MIN`, `0 << -1`). The engine computes integer operators on masked
payloads; these pins hold that a NULL row's masked payloads can never reach
a trap, on both backends. A row with values on both sides still traps.
"""

import sys
from pathlib import Path

import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from test_duckdb_interpreter import _row_schema

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

I64_MIN = -(2**63)


def _serves_as_duckdb(sql, row_schema, rows):
    assert_parity(sql, table(row_schema, rows), expect="AGREE")


def _traps_as_duckdb(sql, row_schema, rows, match):
    assert_parity(sql, table(row_schema, rows), trap=match)


@pytest.mark.parametrize(
    ("sql", "row_schema", "rows"),
    [
        # fuzz seed 22366, shrunk: the NULL left operand's masked 0 minus
        # i64::MIN overflowed.
        (
            "SELECT c2 - c1 AS s FROM __THIS__",
            {"c1": "int", "c2": "int?"},
            [{"c1": I64_MIN, "c2": None}, {"c1": 7, "c2": None}],
        ),
        (
            "SELECT c1 - c2 AS s FROM __THIS__",
            {"c1": "int?", "c2": "int"},
            [{"c1": None, "c2": I64_MIN}],
        ),
        (
            "SELECT c2 - c1 AS s FROM __THIS__",
            {"c1": "int32", "c2": "int32?"},
            [{"c1": -(2**31), "c2": None}],
        ),
        (
            "SELECT c1 << c2 AS s FROM __THIS__",
            {"c1": "int?", "c2": "int"},
            [{"c1": None, "c2": -1}, {"c1": None, "c2": 64}],
        ),
        (
            "SELECT c1 // c2 AS q, c1 % c2 AS r FROM __THIS__",
            {"c1": "int?", "c2": "int"},
            [{"c1": None, "c2": -1}, {"c1": None, "c2": 3}],
        ),
        (
            "SELECT c1 + c2 AS s, c1 * c2 AS m FROM __THIS__",
            {"c1": "int?", "c2": "int?"},
            [{"c1": None, "c2": I64_MIN}, {"c1": I64_MIN, "c2": None}],
        ),
    ],
)
def test_a_null_operand_answers_null(sql, row_schema, rows):
    _serves_as_duckdb(sql, row_schema, rows)


def test_values_on_both_sides_still_trap():
    _traps_as_duckdb(
        "SELECT c2 - c1 AS s FROM __THIS__",
        {"c1": "int", "c2": "int?"},
        [{"c1": I64_MIN, "c2": None}, {"c1": I64_MIN, "c2": 0}],
        match="Overflow in subtraction",
    )


def test_shift_by_a_negative_count_still_traps():
    _traps_as_duckdb(
        "SELECT c1 << c2 AS s FROM __THIS__",
        {"c1": "int?", "c2": "int"},
        [{"c1": None, "c2": -1}, {"c1": 1, "c2": -1}],
        match="negative number",
    )


def test_arrow_path_agrees():
    sql = "SELECT c2 - c1 AS s FROM __THIS__"
    schema = _row_schema({"c1": "int", "c2": "int?"})
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    batch = pa.table({"c1": [I64_MIN], "c2": [None]}, schema=schema)
    assert fn.infer_arrow(batch).column("s").to_pylist() == [None]


# A zero or NULL divisor makes `%` and `//` NULL, but DuckDB still evaluates
# the dividend first, so a trap inside it fires (fuzz seeds 23097, 20523,
# 46043). The rule lives in the lowering, not in a CASE that skipped the
# dividend.
@pytest.mark.parametrize(
    ("sql", "row_schema", "rows", "match"),
    [
        (
            "SELECT (c1 * -3) % c2 AS s FROM __THIS__",
            {"c1": "int32", "c2": "int32?"},
            [{"c1": -(2**31), "c2": None}],
            "Overflow|out of range",
        ),
        (
            "SELECT (c1 * -3) // c2 AS s FROM __THIS__",
            {"c1": "int32", "c2": "int32"},
            [{"c1": -(2**31), "c2": 0}],
            "Overflow|out of range",
        ),
        (
            "SELECT CAST(c3 AS INTEGER) % c2 AS s FROM __THIS__",
            {"c2": "int?", "c3": "str"},
            [{"c2": None, "c3": "O'Brien"}],
            "onver",
        ),
    ],
)
def test_a_zero_or_null_divisor_still_evaluates_the_dividend(
    sql, row_schema, rows, match
):
    _traps_as_duckdb(sql, row_schema, rows, match)


@pytest.mark.parametrize(
    ("sql", "row_schema", "rows"),
    [
        (
            "SELECT c1 % c2 AS r, c1 // c2 AS q FROM __THIS__",
            {"c1": "int?", "c2": "int?"},
            [
                {"c1": 7, "c2": 0},
                {"c1": 7, "c2": None},
                {"c1": None, "c2": 0},
                {"c1": 7, "c2": 2},
                {"c1": I64_MIN, "c2": 0},
            ],
        ),
        (
            "SELECT x // y AS q, x % y AS r FROM __THIS__",
            {"x": "float", "y": "float?"},
            [{"x": 7.5, "y": 0.0}, {"x": 7.5, "y": -0.0}, {"x": 7.5, "y": None}],
        ),
        (
            "SELECT 5 // 0 AS q, 5 % 0 AS r, (5 // 0) || 'a' AS s, c1 FROM __THIS__",
            {"c1": "int"},
            [{"c1": 1}],
        ),
    ],
)
def test_a_zero_or_null_divisor_answers_null(sql, row_schema, rows):
    _serves_as_duckdb(sql, row_schema, rows)
