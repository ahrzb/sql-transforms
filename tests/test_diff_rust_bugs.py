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

from differential import check, rows


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
