"""DECIMAL expressions agree with DuckDB (docs/specs/decimal-expressions.md).

A numeric literal with a point is a DECIMAL on DuckDB, and so is everything
computed from it by `+ - * %`, the casts, and the rounding builtins. Each
case here is the campaign's own verdict (`fuzz.parity`): rows, types, and
traps against the optimizer-off oracle.
"""

import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity  # noqa: E402

ROWS = pa.table(
    {
        "a": pa.array([3, -5, None, 2147483647, 0, -7], pa.int32()),
        "b": pa.array([9223372036854775807, -2, None, 7, 1, 0], pa.int64()),
        "t": pa.array([127, -128, None, 1, 0, 5], pa.int8()),
        "f": pa.array([0.5, None, -2.5, 1e300, 0.0, 3.25], pa.float64()),
    }
)


def _q(expr: str) -> str:
    return f"SELECT {expr} AS x FROM __THIS__"


# -------------------------------------------------------------- the literal
# §1: DECIMAL(digits, digits after the point), leading zeros counted.


@pytest.mark.parametrize("lit", ["2.5", "-2.5", "0.1", ".5", "1.", "0.05", "12345.678"])
def test_a_decimal_literal_is_duckdbs_decimal(lit):
    assert_parity(_q(lit), ROWS, expect="AGREE")


def test_the_nightly_rounding_finding():
    # Seed 1119492: CAST of a DECIMAL rounds half away from zero; a DOUBLE
    # (CAST(-2.5e0 ...)) rounds half to even. Both agree now.
    sql = "SELECT CAST(-2.5 AS BIGINT) AS x, CAST(-2.5e0 AS BIGINT) AS y FROM __THIS__"
    assert_parity(sql, ROWS, expect="AGREE")


def test_decimal_equality_is_exact():
    # 0.1 + 0.2 = 0.3 is true on DuckDB, false in DOUBLE.
    assert_parity(_q("0.1 + 0.2 = 0.3"), ROWS, expect="AGREE")
    assert_parity(_q("a * 0.1 = 0.3"), ROWS, expect="AGREE")


# ------------------------------------------------------------ arithmetic
# §3-§5, integers joining as DECIMAL(3/5/10/19, 0).


@pytest.mark.parametrize(
    "expr",
    [
        "a * 0.5",
        "a + 2.5",
        "a - 2.5",
        "2.5 - a",
        "a % 2.5",
        "2.5 % a",
        "b * 0.5",
        "b + 2.5",
        "t * 2.5",
        "2.5 * 3",
        "2.5 * 1.5 * 2",
        "0.5 - 0.25",
        "-(a * 0.5)",
        "+(2.5)",
        "a * 0.5 * 0.5 * 0.5",
    ],
)
def test_decimal_arithmetic_keeps_duckdbs_type(expr):
    assert_parity(_q(expr), ROWS, expect="AGREE")


@pytest.mark.parametrize(
    "expr", ["a / 2.5", "a // 2.5", "2.5 / 0.0", "f * 2.5", "f + 0.1", "2.5 + 1.0e0"]
)
def test_division_and_double_mixes_are_double(expr):
    assert_parity(_q(expr), ROWS, expect="AGREE")


@pytest.mark.parametrize(
    "expr, trap",
    [
        # Capped at 18 digits: the storage bound traps.
        ("99999999999999999.9 + a", r"Overflow in addition of DECIMAL\(18\)"),
        ("9999999999999.99999 * a", r"Overflow in multiplication of DECIMAL\(18\)"),
        # Capped at 38.
        ("a * 1.0000000000000000000000000000000000001", r"DECIMAL\(38\)"),
        ("b * 12345678901234567890.123", r"DECIMAL\(38\)"),
    ],
)
def test_a_capped_width_traps_like_duckdb(expr, trap):
    assert_parity(_q(expr), ROWS, trap=trap)


def test_a_constant_null_operand_is_sqlnull():
    sql = (
        "SELECT 2.5 + NULL AS x, NULL * 2.5 AS y, "
        "-(CAST(NULL AS DECIMAL(3,1))) AS z, 2.5 + CAST(NULL AS INTEGER) AS w "
        "FROM __THIS__"
    )
    assert_parity(sql, ROWS, expect="AGREE")


@pytest.mark.parametrize(
    "expr",
    [
        # SQLNULL re-promotes by signature: unary minus makes it BIGINT,
        # COALESCE and CASE skip it when unifying (campaign seeds 4166,
        # 6442, 10376, 15873).
        "- (NULL * 75.129)",
        "struct_pack(f0 := (- (-0.5 * NULL)))",
        "coalesce((1.75 + 41.724), (NULL - 1.75))",
        "coalesce((NULL * a), 0.75)",
        "CASE WHEN a > 0 THEN 2.5 ELSE NULL % 0.5 END",
    ],
)
def test_a_decimal_operator_over_null_is_an_adoptable_null(expr):
    assert_parity(_q(expr), ROWS, expect="AGREE")


def test_a_decimal_remainder_by_zero_is_null():
    assert_parity(_q("a % 0.0"), ROWS, expect="AGREE")


def test_decimal_bitwise_is_a_binder_error():
    assert_parity(_q("2.5 & 1"), ROWS, expect="REFUSED")


def test_a_scale_past_38_refuses_as_duckdb_does():
    sql = _q("0.00000000000000000001 * 0.00000000000000000001")
    assert_parity(sql, ROWS, expect="REFUSED")


# ---------------------------------------------------------------- casts
# §8: half away from zero, checked against the target.


@pytest.mark.parametrize(
    "expr",
    [
        "CAST(a * 2.5 AS INTEGER)",
        "CAST(a * 0.25 AS BIGINT)",
        "CAST(a * 0.25 AS DECIMAL(9,1))",
        "CAST(a * 2.5 AS VARCHAR)",
        "CAST(-0.05 AS VARCHAR)",
        "CAST(2.50 AS VARCHAR)",
        "CAST(t AS DECIMAL(4,1))",
        "CAST(a * 0.25 AS DOUBLE)",
        "CAST(-0.05 AS DECIMAL(2,1))",
        "CAST(2.5 AS DECIMAL(4,0))",
        "CAST(NULL AS DECIMAL(4,1))",
    ],
)
def test_a_decimal_cast_agrees(expr):
    assert_parity(_q(expr), ROWS, expect=("AGREE", "AGREE_TRAP"))


@pytest.mark.parametrize(
    "expr, trap",
    [
        ("CAST(a * 2.5 AS TINYINT)", r"Failed to cast decimal value"),
        ("CAST(a AS DECIMAL(4,1))", r"Could not cast value"),
        ("CAST(a * 0.25 AS DECIMAL(3,0))", r"failed: value is out of range"),
        ("CAST(b AS DECIMAL(4,2))", r"Could not cast value"),
    ],
)
def test_a_failing_decimal_cast_traps_like_duckdb(expr, trap):
    assert_parity(_q(expr), ROWS, trap=trap)


@pytest.mark.parametrize(
    "expr",
    [
        "TRY_CAST(a * 2.5 AS TINYINT)",
        "TRY_CAST(a AS DECIMAL(3,1))",
        "TRY_CAST(a * 0.25 AS DECIMAL(4,1))",
    ],
)
def test_a_failing_try_cast_is_null(expr):
    assert_parity(_q(expr), ROWS, expect="AGREE")


# ----------------------------------------------- comparison, unification
# §7: the common DECIMAL, or DOUBLE against a DOUBLE.


@pytest.mark.parametrize(
    "expr",
    [
        "a * 0.5 > 1",
        "0.5 < f",
        "2.5 = 2.50",
        "a IN (2.50, 3)",
        "a BETWEEN 0.5 AND 2.5",
        "f IN (1.0, 0.5)",
        "CASE WHEN a > 0 THEN 2.5 ELSE 10.25 END",
        "CASE WHEN a > 0 THEN 2.5 ELSE a END",
        "coalesce(2.5, b)",
        "greatest(2.5, 1.25, a)",
        "least(0.5, f)",
        "struct_pack(p := 1.5, q := a * 0.5)",
    ],
)
def test_a_decimal_unifies_like_duckdb(expr):
    assert_parity(_q(expr), ROWS, expect="AGREE")


# -------------------------------------------------------------- builtins
# numeric.cpp: abs/ceil/floor/round/trunc have DECIMAL overloads; the rest
# read a DECIMAL as DOUBLE.


@pytest.mark.parametrize(
    "expr",
    [
        "abs(a * -0.5)",
        "floor(a * 0.25)",
        "ceil(a * 0.25)",
        "ceiling(2.01)",
        "round(a * 0.25)",
        "trunc(a * 0.25)",
        "floor(-2.5)",
        "round(-2.5)",
        "round(a * 0.125, 2)",
        "round(a * 0.125, 0)",
        "round(a * 0.125, -1)",
        "round(a * 0.125, 5)",
        "round(a * 0.125, -30)",
        "trunc(a * 0.125, 2)",
        "trunc(a * 0.125, -1)",
        "floor(CAST(NULL AS DECIMAL(4,2)))",
        "ln(2.5)",
        "sqrt(a * 0.5)",
        "log(a, 2.0)",
        "pow(2.5, 2)",
    ],
)
def test_a_builtin_over_a_decimal_agrees(expr):
    assert_parity(_q(expr), ROWS, expect=("AGREE", "AGREE_TRAP"))


def test_round_with_a_column_precision_is_duckdbs_not_implemented():
    assert_parity(_q("round(2.5, a)"), ROWS, expect="REFUSED")
