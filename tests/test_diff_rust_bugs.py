"""Rust-engine bugs: cases where `infer` disagrees with `transform`.

Each of these violates the README's promise that both paths return identical
values -- a user gets a different answer at serving time than at batch time.
Codegen matches the DataFusion oracle and PASSES; the Rust engine is xfail-ed
(strict) pending its BACKLOG ticket. If one starts XPASSing, the Rust bug was
fixed: delete the rust_bug() call and close the ticket.

Surface Rust rejects outright (unary minus, `||`) is a different class of gap --
ticketed, but not testable here since both engines decline it.
"""

from __future__ import annotations

from differential import check, check_both_raise, rows


def test_cast_float_to_varchar_keeps_the_decimal_point(rust_bug):
    rust_bug(
        "Rust bug: expr::display_value uses f64::to_string, rendering 1.0 as '1'; "
        "DataFusion gives '1.0'. transform/infer return different values."
    )
    check(
        "SELECT CAST(f AS VARCHAR) AS x FROM t",
        {"t": rows({"f": "float"}, [{"f": 1.0}])},
        expect=[{"x": "1.0"}],
    )


def test_round_of_an_int_returns_a_float(rust_bug):
    rust_bug(
        "Rust bug: eval_builtin's ROUND passes an int through unchanged, returning "
        "int 3; DataFusion returns float 3.0. transform/infer differ in output TYPE."
    )
    check(
        "SELECT ROUND(a) AS x FROM t",
        {"t": rows({"a": "int"}, [{"a": 3}])},
        expect=[{"x": 3.0}],
    )


def test_nullif_compares_numerically_across_int_and_float(rust_bug):
    rust_bug(
        "Rust bug: NULLIF uses Value's variant-tagged PartialEq, so Int(1) != "
        "Float(1.0) and it returns 1; DataFusion coerces and returns NULL."
    )
    check(
        "SELECT NULLIF(a, 1.0) AS x FROM t",
        {"t": rows({"a": "int"}, [{"a": 1}])},
        expect=[{"x": None}],
    )


def test_substr_start_below_one_uses_postgres_windowing(rust_bug):
    rust_bug(
        "Rust bug: expr::substr clamps a start <= 0 to the beginning and keeps the "
        "full length, so SUBSTR('hello',0,3) is 'hel'; DataFusion windows [start, "
        "start+length) so positions < 1 consume the length -> 'he'."
    )
    check(
        "SELECT SUBSTR(s, 0, 3) AS x FROM t",
        {"t": rows({"s": "str"}, [{"s": "hello"}])},
        expect=[{"x": "he"}],
    )


def test_nan_comparison_returns_a_bool(rust_bug):
    rust_bug(
        "Rust bug: compare_values raises on NaN; DataFusion has a total order "
        "where NaN sorts below every non-NaN value, so NaN < 1.0 is True."
    )
    check(
        "SELECT (z / z < o) AS x FROM t",
        {"t": rows({"z": "float", "o": "float"}, [{"z": 0.0, "o": 1.0}])},
        expect=[{"x": True}],
    )


def test_cast_str_to_bool_accepts_the_full_token_set(rust_bug):
    rust_bug(
        "Rust bug: eval_cast only accepts 'true' for str->bool; DataFusion accepts "
        "t/1/yes/y/on (and f/0/no/n/off), case-insensitive. CAST('t' AS BOOL)=True."
    )
    check(
        "SELECT CAST(s AS BOOLEAN) AS x FROM t",
        {"t": rows({"s": "str"}, [{"s": "t"}])},
        expect=[{"x": True}],
    )


def test_cast_str_to_int_rejects_surrounding_whitespace(rust_bug):
    rust_bug(
        "Rust bug: eval_cast str->int calls str::trim, so CAST(' 42 ' AS BIGINT) "
        "returns 42; DataFusion rejects surrounding whitespace and errors."
    )
    check_both_raise(
        "SELECT CAST(s AS BIGINT) AS x FROM t",
        {"t": rows({"s": "str"}, [{"s": " 42 "}])},
    )
