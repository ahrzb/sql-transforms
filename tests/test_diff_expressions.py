"""Differential coverage of the Rust engine's scalar expression surface."""

import pytest
from differential import check, rows


@pytest.mark.parametrize(
    "a, b, q",
    [
        (6, 2, 3),
        (7, 2, 3),  # int/int truncates toward zero
        (-7, 2, -3),
    ],
)
def test_int_division_truncates(a, b, q):
    check(
        "SELECT a / b AS q FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": a, "b": b}])},
        expect=[{"q": q}],
    )


@pytest.mark.parametrize(
    "a, b",
    [(6, 2), (7, 2), (-7, 2), (5, 3)],
)
def test_int_mod(a, b):
    check(
        "SELECT a % b AS r FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": a, "b": b}])},
    )


@pytest.mark.parametrize(
    "a, b",
    [(3, 2.0), (7, 4.0)],  # int / float promotes to float
)
def test_mixed_int_float_promotes(a, b):
    check(
        "SELECT a / b AS q FROM t",
        {"t": rows({"a": "int", "b": "float"}, [{"a": a, "b": b}])},
    )


@pytest.mark.parametrize(
    "x, y, expected",
    [(1, 2, True), (2, 1, False), (2, 2, False)],
)
def test_comparison(x, y, expected):
    check(
        "SELECT x < y AS lt FROM t",
        {"t": rows({"x": "int", "y": "int"}, [{"x": x, "y": y}])},
        expect=[{"lt": expected}],
    )


@pytest.mark.parametrize(
    "p, q, expected",
    [
        (True, False, False),
        (True, None, None),  # three-valued: TRUE AND NULL = NULL
        (False, None, False),  # FALSE AND NULL = FALSE
    ],
)
def test_and_three_valued(p, q, expected):
    check(
        "SELECT p AND q AS r FROM t",
        {"t": rows({"p": "bool?", "q": "bool?"}, [{"p": p, "q": q}])},
        expect=[{"r": expected}],
    )


@pytest.mark.parametrize(
    "val, expected",
    [(3.7, 3), (-3.7, -3)],  # float->int truncates toward zero
)
def test_cast_float_to_int(val, expected):
    check(
        "SELECT CAST(x AS BIGINT) AS c FROM t",
        {"t": rows({"x": "float"}, [{"x": val}])},
        expect=[{"c": expected}],
    )


def test_string_builtins():
    check(
        "SELECT UPPER(s) AS u, LOWER(s) AS l, TRIM(s) AS t2 FROM t",
        {"t": rows({"s": "str"}, [{"s": " AbC "}])},
    )


@pytest.mark.parametrize(
    "a, b",
    [(1, 2), (None, 2), (1, None)],  # NULL propagates through arithmetic
)
def test_null_propagation(a, b):
    check(
        "SELECT a + b AS s FROM t",
        {"t": rows({"a": "int?", "b": "int?"}, [{"a": a, "b": b}])},
    )


def test_coalesce_and_nullif():
    check(
        "SELECT COALESCE(a, b) AS c, NULLIF(a, b) AS n FROM t",
        {
            "t": rows(
                {"a": "int?", "b": "int?"},
                [{"a": None, "b": 5}, {"a": 3, "b": 3}, {"a": 7, "b": 2}],
            )
        },
    )


def test_abs_and_round():
    check(
        "SELECT ABS(x) AS a, ROUND(y) AS r FROM t",
        {"t": rows({"x": "int", "y": "float"}, [{"x": -4, "y": 2.6}])},
    )
