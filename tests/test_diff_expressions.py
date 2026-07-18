"""Differential coverage of the Rust engine's scalar expression surface."""

import pyarrow as pa
import pytest
from differential import _rows_equal, check, rows

from sql_transform import SQLTransform


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


def test_window_aggregate_over_expression_parity():
    train = pa.table({"x": [1.0, 2.0, 3.0, 4.0]})
    t = SQLTransform("SELECT x / AVG(x + 1) OVER () AS r FROM __THIS__").fit(train)
    batch_out = t.transform(train).to_pylist()
    infer_out = [r.model_dump() for r in t.infer_batch(train.to_pylist())]
    assert _rows_equal(batch_out, infer_out)
    # AVG(x+1) over [2,3,4,5] = 3.5
    assert abs(batch_out[0]["r"] - (1.0 / 3.5)) < 1e-9


def test_window_aggregate_quoted_case_sensitive_column_plain_parity():
    train = pa.table({"Age": [10.0, 20.0, 30.0, 40.0]})
    t = SQLTransform('SELECT "Age" / AVG("Age") OVER () AS r FROM __THIS__').fit(train)
    batch_out = t.transform(train).to_pylist()
    infer_out = [r.model_dump() for r in t.infer_batch(train.to_pylist())]
    assert _rows_equal(batch_out, infer_out)
    # AVG([10,20,30,40]) = 25.0
    assert abs(batch_out[0]["r"] - (10.0 / 25.0)) < 1e-9


def test_window_aggregate_quoted_case_sensitive_column_expression_parity():
    train = pa.table({"Age": [10.0, 20.0, 30.0, 40.0]})
    t = SQLTransform('SELECT "Age" / AVG("Age" + 1) OVER () AS r FROM __THIS__').fit(
        train
    )
    batch_out = t.transform(train).to_pylist()
    infer_out = [r.model_dump() for r in t.infer_batch(train.to_pylist())]
    assert _rows_equal(batch_out, infer_out)
    # AVG([11,21,31,41]) = 26.0
    assert abs(batch_out[0]["r"] - (10.0 / 26.0)) < 1e-9


def test_window_aggregate_unquoted_case_sensitive_column_errors():
    # DataFusion folds unquoted `Age` to `age`; with only `Age` in the table
    # this errors, matching DataFusion's own case-folding behavior.
    train = pa.table({"Age": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        SQLTransform("SELECT Age / AVG(Age) OVER () AS r FROM __THIS__").fit(train)


@pytest.mark.parametrize(
    "value, rendered",
    [
        (1e-5, "0.00001"),  # DataFusion fixed one decade below Python's threshold
        (1.5e-5, "0.000015"),
    ],
)
def test_cast_float_to_varchar_small_decimals(value, rendered):
    # The [1e-5, 1e-4) band where codegen was uniquely wrong (Python used
    # scientific notation) and Rust already matched DataFusion -- so it runs
    # clean on BOTH backends. Codegen's full float formatting (exponent form,
    # integer-valued, large) is pinned by runtime_test; those other shapes also
    # trip Rust's f64::to_string bug and are covered by the xfail-on-rust cases.
    check(
        "SELECT CAST(f AS VARCHAR) AS x FROM t",
        {"t": rows({"f": "float"}, [{"f": value}])},
        expect=[{"x": rendered}],
    )


@pytest.mark.parametrize(
    "a, expr, expected",
    [
        (9223372036854775807, "a * 2", -2),
        (9223372036854775807, "a + 1", -9223372036854775808),
        (-9223372036854775808, "a - 1", 9223372036854775807),
    ],
)
def test_integer_arithmetic_wraps_at_i64(a, expr, expected):
    # Codegen used Python bigints and overflowed; DataFusion (and Rust) wrap i64.
    check(
        f"SELECT {expr} AS x FROM t",
        {"t": rows({"a": "int"}, [{"a": a}])},
        expect=[{"x": expected}],
    )
