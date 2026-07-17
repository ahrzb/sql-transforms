"""Unit tests for the codegen semantics runtime.

Each test here pins a place where naive Python diverges from the Rust engine
(src/expr.rs). See docs/superpowers/specs/2026-07-17-codegen-inferfn-design.md.
"""

import math

import pytest

from sql_transform._codegen import runtime as rt


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


def test_comparison_propagates_null():
    assert rt.eq(None, 1) is None
    assert rt.lt(1, None) is None


def test_comparison_across_int_and_float():
    assert rt.eq(1, 1.0) is True
    assert rt.lt(1, 1.5) is True
    assert rt.gte(2.0, 2) is True


def test_comparison_of_strings_and_bools():
    assert rt.lt("a", "b") is True
    assert rt.eq("a", "a") is True
    assert rt.lt(False, True) is True


def test_comparison_of_mismatched_types_errors():
    # Rust falls through to as_f64, which errors on a string.
    with pytest.raises(ValueError, match="arithmetic"):
        rt.eq("a", 1)


def test_comparison_of_nan_errors():
    with pytest.raises(ValueError, match="NaN"):
        rt.lt(math.nan, 1.0)


def test_kleene_and():
    assert rt.and_(True, True) is True
    assert rt.and_(True, False) is False
    assert rt.and_(False, None) is False  # false dominates
    assert rt.and_(True, None) is None
    assert rt.and_(None, None) is None


def test_kleene_or():
    assert rt.or_(False, False) is False
    assert rt.or_(True, None) is True  # true dominates
    assert rt.or_(False, None) is None
    assert rt.or_(None, None) is None


def test_not_is_tri_valued():
    assert rt.not_(True) is False
    assert rt.not_(False) is True
    assert rt.not_(None) is None


def test_logic_rejects_non_bool_even_when_other_side_decides():
    # Rust's logic() calls as_tribool on BOTH operands before matching, so this
    # errors rather than short-circuiting to False.
    with pytest.raises(ValueError, match="boolean"):
        rt.and_(False, 1)
    with pytest.raises(ValueError, match="boolean"):
        rt.or_(True, "x")


def test_join_eq_is_type_strict_and_null_never_matches():
    assert rt.join_eq(1, 1) is True
    assert rt.join_eq(1, 1.0) is False  # Value::Int(1) != Value::Float(1.0)
    assert rt.join_eq(None, None) is False  # NULL keys never match
    assert rt.join_eq(None, 1) is False


def test_casts_propagate_null():
    assert rt.cast_int(None) is None
    assert rt.cast_str(None) is None


def test_cast_int_truncates_floats():
    assert rt.cast_int(1.9) == 1
    assert rt.cast_int(-1.9) == -1


def test_cast_int_parses_strings_and_errors_on_junk():
    assert rt.cast_int(" 42 ") == 42
    with pytest.raises(ValueError, match="Cannot cast"):
        rt.cast_int("abc")


def test_cast_bool_from_string_is_case_insensitive_true():
    assert rt.cast_bool("TRUE") is True
    assert rt.cast_bool("yes") is False


def test_cast_float_and_bool_from_numbers():
    assert rt.cast_float(1) == 1.0
    assert rt.cast_float(True) == 1.0
    assert rt.cast_bool(0) is False
    assert rt.cast_bool(2) is True


def test_cast_str_uses_datafusion_float_formatting():
    # Measured: DataFusion's CAST(1.0 AS VARCHAR) is '1.0'. Rust gives '1' -- a
    # Rust bug (see the DECIDED section in the design doc: codegen matches the
    # DataFusion oracle, not Rust). cast_str delegates to `display`, which is
    # already DataFusion-parity (test_display_matches_datafusion pins this).
    assert rt.cast_str(1.0) == "1.0"


def test_round_is_half_away_from_zero_not_bankers():
    # Python's round(0.5) == 0 and round(2.5) == 2 (banker's); Rust rounds away.
    assert rt.round_(0.5) == 1.0
    assert rt.round_(2.5) == 3.0
    assert rt.round_(-2.5) == -3.0
    assert type(rt.round_(2.5)) is float  # must stay a float, not become an int


def test_round_of_an_int_returns_a_float():
    # Measured: DataFusion's ROUND(3) is 3.0. Rust returns int 3 -- a Rust bug.
    assert rt.round_(3) == 3.0
    assert type(rt.round_(3)) is float


def test_abs_preserves_its_argument_type():
    assert rt.abs_(-3) == 3
    assert type(rt.abs_(-3)) is int
    assert rt.abs_(-3.5) == 3.5


def test_abs_and_round_reject_non_numbers():
    with pytest.raises(ValueError, match="ABS"):
        rt.abs_("x")
    with pytest.raises(ValueError, match="ROUND"):
        rt.round_("x")


def test_string_builtins_propagate_null():
    assert rt.upper(None) is None
    assert rt.trim(None) is None
    assert rt.substr(None, 1) is None
    assert rt.substr("abc", None) is None


def test_string_builtins():
    assert rt.upper("aB") == "AB"
    assert rt.lower("aB") == "ab"
    assert rt.trim("  x  ") == "x"


def test_substr_is_one_indexed_and_clamps():
    assert rt.substr("hello", 2) == "ello"
    assert rt.substr("hello", 2, 3) == "ell"
    assert rt.substr("hello", 0) == "hello"  # start <= 0 clamps to the beginning
    assert rt.substr("hello", 2, 99) == "ello"  # length clamps to the end
    assert rt.substr("hello", 99) == ""


def test_concat_skips_nulls_and_never_returns_null():
    assert rt.concat("a", None, "b") == "ab"
    assert rt.concat(None) == ""
    assert rt.concat("n=", 1) == "n=1"


def test_coalesce_returns_first_non_null():
    assert rt.coalesce(None, None, 3) == 3
    assert rt.coalesce(None, None) is None


def test_nullif_compares_numerically_across_int_and_float():
    assert rt.nullif(1, 1) is None
    assert rt.nullif(1, 2) == 1
    # Measured: DataFusion nulls this. Rust returns 1 (variant-tagged eq) -- a bug.
    assert rt.nullif(1, 1.0) is None
    assert rt.nullif(None, None) is None
    assert rt.nullif(None, 1) is None  # a[0] is NULL, so the result is NULL
    assert rt.nullif("a", "b") == "a"
