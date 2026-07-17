"""Unit tests for the codegen semantics runtime.

Each test here pins a place where naive Python diverges from the Rust engine
(src/expr.rs). See docs/superpowers/specs/2026-07-17-codegen-inferfn-design.md.
"""

import math

import pytest

from sql_transform import _codegen_runtime as rt


def test_int_division_truncates_toward_zero():
    # Python's -7 // 2 == -4 (floor); Rust's -7 / 2 == -3 (truncate).
    assert rt.div(-7, 2) == -3
    assert rt.div(7, 2) == 3
    assert rt.div(-7, -2) == 3


def test_int_mod_takes_sign_of_dividend():
    # Python's -7 % 2 == 1 (sign of divisor); Rust's == -1 (sign of dividend).
    assert rt.mod(-7, 2) == -1
    assert rt.mod(7, -2) == 1


def test_int_div_and_mod_by_zero_raise():
    with pytest.raises(ValueError, match="division by zero"):
        rt.div(1, 0)
    with pytest.raises(ValueError, match="division by zero"):
        rt.mod(1, 0)


def test_float_division_by_zero_is_ieee_not_an_error():
    # Rust yields inf/nan; Python's / would raise ZeroDivisionError.
    assert rt.div(1.0, 0.0) == math.inf
    assert rt.div(-1.0, 0.0) == -math.inf
    assert math.isnan(rt.div(0.0, 0.0))


def test_float_mod_is_c_style():
    assert rt.mod(-7.0, 2.0) == math.fmod(-7.0, 2.0)
    assert math.isnan(rt.mod(1.0, 0.0))


def test_arithmetic_propagates_null():
    assert rt.add(None, 1) is None
    assert rt.add(1, None) is None
    assert rt.mul(None, None) is None


def test_int_int_stays_int_mixed_promotes_to_float():
    assert rt.add(1, 2) == 3
    assert type(rt.add(1, 2)) is int
    assert type(rt.add(1, 2.0)) is float


def test_bool_is_not_a_number():
    # Python would give True + 1 == 2; Rust's as_f64 errors on a bool.
    with pytest.raises(ValueError, match="arithmetic"):
        rt.add(True, 1)


def test_string_is_not_a_number():
    with pytest.raises(ValueError, match="arithmetic"):
        rt.add("a", 1)


def test_tags_keep_int_float_bool_distinct():
    # Value's Eq/Hash are variant-tagged: Int(1) != Float(1.0), True is not 1.
    assert rt.tag(1) != rt.tag(1.0)
    assert rt.tag(True) != rt.tag(1)
    assert rt.key(1) != rt.key(1.0)
    assert not rt.val_eq(1, 1.0)
    assert not rt.val_eq(True, 1)
    assert rt.val_eq(1, 1)
    assert rt.val_eq(None, None)


def test_truthy_is_strict_bool_true():
    # RelNode::Filter keeps a row only on Value::Bool(true).
    assert rt.truthy(True)
    assert not rt.truthy(False)
    assert not rt.truthy(None)
    assert not rt.truthy(1)
    assert not rt.truthy("yes")


@pytest.mark.parametrize(
    "value, expected",
    [
        # Every expectation below was MEASURED against DataFusion, not reasoned.
        (1.0, "1.0"),  # Rust gives "1" here -- a Rust bug, xfail-ed in Task 11
        (1.5, "1.5"),
        (0.1, "0.1"),
        (1e300, "1e300"),  # Python str() gives "1e+300"; Rust gives 300 digits
        (math.nan, "NaN"),  # Python str() gives "nan"
        (math.inf, "inf"),
        (-math.inf, "-inf"),
        (None, ""),
        (True, "true"),
        (False, "false"),
        (42, "42"),
        ("hi", "hi"),
    ],
)
def test_display_matches_datafusion(value, expected):
    assert rt.display(value) == expected
