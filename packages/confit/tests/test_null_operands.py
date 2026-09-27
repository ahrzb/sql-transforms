"""A NULL operand makes an integer operator NULL; it never makes it trap.

DuckDB answers NULL for `x op NULL` whatever `x` is, even where the same
operator with a value in NULL's place would overflow or reject its operand
(`0 - i64::MIN`, `0 << -1`). The engine computes integer operators on masked
payloads; these pins hold that a NULL row's masked payloads can never reach
a trap, on both backends. A row with values on both sides still traps.
"""

import os

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, compare
from confit.oracle import Oracle
from test_duckdb_interpreter import _row_schema, static

I64_MIN = -(2**63)


def _confit(sql, row_schema, rows, force_interp):
    prev = os.environ.pop("SPECIALIZER_FORCE_INTERP", None)
    try:
        if force_interp:
            os.environ["SPECIALIZER_FORCE_INTERP"] = "1"
        fn = DuckDBInferFn(
            sql, row_tables={"__THIS__": _row_schema(row_schema)}, static_tables={}
        )
    finally:
        os.environ.pop("SPECIALIZER_FORCE_INTERP", None)
        if prev is not None:
            os.environ["SPECIALIZER_FORCE_INTERP"] = prev
    assert fn.backend == ("interpreter" if force_interp else "cranelift")
    return fn.infer_rows(rows)


def _oracle(sql, row_schema, rows):
    o = Oracle()
    o.load("__THIS__", static(row_schema, rows))
    return compare.rows(o.answer(sql))


def _serves_as_duckdb(sql, row_schema, rows):
    want = _oracle(sql, row_schema, rows)
    for force in (False, True):
        compare.assert_rows(_confit(sql, row_schema, rows, force), want, ctx=sql)


def _traps_as_duckdb(sql, row_schema, rows, match):
    with pytest.raises(Exception, match=match):
        _oracle(sql, row_schema, rows)
    for force in (False, True):
        with pytest.raises(Exception, match=match):
            _confit(sql, row_schema, rows, force)


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
