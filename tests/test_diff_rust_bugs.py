"""Pinning tests for the Rust InferFn parity bugs (transform != infer).

Each divergence documented in docs/BACKLOG.md ("Rust engine (InferFn) parity
bugs") is pinned here as a strict xfail-on-rust differential test: DataFusion is
the oracle, so `check`/`check_both_raise` currently FAIL because the Rust engine
disagrees. When a bug is fixed the test passes -> strict xfail turns XPASS into a
failure -> forces removing the marker. Remove each marker as its fix lands.
"""

from differential import check, check_both_raise, row, rows
import pytest

_R = "__THIS__"


@pytest.mark.xfail(strict=True, reason="#1 display_value float uses f64::to_string")
def test_float_display_cast_and_concat():
    check("SELECT CAST(x AS VARCHAR) AS s FROM __THIS__", {_R: row(x=1.0)})
    check("SELECT CAST(x AS VARCHAR) AS s FROM __THIS__", {_R: row(x=1e300)})
    check("SELECT CONCAT('v', x) AS s FROM __THIS__", {_R: row(x=1.0)})


def test_round_int_returns_float():
    check("SELECT ROUND(x) AS r FROM __THIS__", {_R: row(x=3)})


def test_nullif_numeric_coercion():
    check("SELECT NULLIF(1, 1.0) AS n FROM __THIS__", {_R: row(z=0)})


def test_unary_minus():
    check("SELECT -a AS m FROM __THIS__", {_R: row(a=5)})
    check("SELECT -a AS m FROM __THIS__", {_R: row(a=2.5)})


def test_string_concat_operator():
    check("SELECT a || '!' AS s FROM __THIS__", {_R: rows({"a": "str"}, [{"a": "hi"}])})
    check("SELECT a || NULL AS s FROM __THIS__", {_R: rows({"a": "str"}, [{"a": "hi"}])})
    check("SELECT a || 5 AS s FROM __THIS__", {_R: rows({"a": "str"}, [{"a": "hi"}])})


def test_coalesce_numeric_supertype():
    check("SELECT COALESCE(3, 9.0) AS c FROM __THIS__", {_R: row(z=0)})


@pytest.mark.xfail(strict=True, reason="#7 SUBSTR start<=0 not Postgres windowing")
def test_substr_nonpositive_start():
    check("SELECT SUBSTR(s, 0, 3) AS r FROM __THIS__", {_R: rows({"s": "str"}, [{"s": "hello"}])})
    check("SELECT SUBSTR(s, -2, 5) AS r FROM __THIS__", {_R: rows({"s": "str"}, [{"s": "hello"}])})


@pytest.mark.xfail(strict=True, reason="#8 NaN = NaN raises in infer, DF returns True")
def test_nan_equals_nan():
    check(
        "SELECT (CAST('NaN' AS DOUBLE) = CAST('NaN' AS DOUBLE)) AS b FROM __THIS__",
        {_R: row(z=0)},
    )


@pytest.mark.xfail(strict=True, reason="#9 CAST(str AS BOOL) accepts only 'true'")
def test_cast_string_to_bool():
    check("SELECT CAST('t' AS BOOLEAN) AS b FROM __THIS__", {_R: row(z=0)})
    check("SELECT CAST('yes' AS BOOLEAN) AS b FROM __THIS__", {_R: row(z=0)})
    check("SELECT CAST('1' AS BOOLEAN) AS b FROM __THIS__", {_R: row(z=0)})


@pytest.mark.xfail(strict=True, reason="#10 CAST(' 42 ' AS INT) strips whitespace, DF errors")
def test_cast_whitespace_string_to_number():
    check_both_raise("SELECT CAST(' 42 ' AS BIGINT) AS n FROM __THIS__", {_R: row(z=0)})
    check_both_raise("SELECT CAST(' 4.5 ' AS DOUBLE) AS n FROM __THIS__", {_R: row(z=0)})
