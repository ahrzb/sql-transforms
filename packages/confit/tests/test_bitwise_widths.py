"""Bitwise operators at a narrow width agree with DuckDB.

DuckDB computes `<< >> & | xor` at the operands' own width: an integer
literal adapts to its partner like it does for `+` (`tiny >> 8` is
TINYINT), and a right shift by the width or more is 0 there, where an i64
shift would keep the sign bits (`(-1)::TINYINT >> 8` is 0).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

ROWS = table(
    {"t": "int8", "s": "int16", "i": "int32", "b": "int", "y": "int8"},
    [
        {"t": -1, "s": 3, "i": -1, "b": -1, "y": 8},
        {"t": -128, "s": -7, "i": 5, "b": 7, "y": 7},
        {"t": 100, "s": 1, "i": -2147483648, "b": 1, "y": 40},
    ],
)


@pytest.mark.parametrize(
    "expr",
    [
        "t >> y",
        "i >> y",
        "b >> y",
        "t >> 8",
        "t >> 300",
        "t >> -1",
        "i >> 32",
        "CAST(-1 AS TINYINT) >> 8",
        "(t >> y) IN (0, -1)",
        "8 >> t",
        "t & 3",
        "t | 3 | s",
        "1 << 2",
        "b & 3",
    ],
)
def test_a_narrow_bitwise_op_agrees(expr):
    assert_parity(f"SELECT {expr} AS o FROM __THIS__", ROWS, expect="AGREE")


@pytest.mark.parametrize("expr", ["t << 1", "s << 2", "i << 31"])
def test_a_narrow_left_shift_traps_where_duckdb_does(expr):
    assert_parity(f"SELECT {expr} AS o FROM __THIS__", ROWS, expect="AGREE_TRAP")
