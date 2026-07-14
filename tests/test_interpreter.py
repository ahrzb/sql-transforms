"""Tests for the Rust SQL interpreter (sql_transform._interpreter).

Every behavioral test compares InferFn output against real DataFusion
batch output for the same SQL + data, per the Phase 2 spec's testing
strategy.
"""

import datafusion
from sql_transform._interpreter import InferFn


def test_module_imports_and_constructs():
    fn = InferFn("SELECT age FROM data", row_tables=["data"], static_tables={})
    assert fn is not None


def _expected(sql: str, data: dict) -> list[dict]:
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name="data")
    return ctx.sql(sql).collect()[0].to_pylist()


def test_column_pass_through():
    sql = "SELECT age FROM data"
    data = {"age": [30]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 30}]})
    assert actual == _expected(sql, data)


def test_multiple_columns():
    sql = "SELECT a, b FROM data"
    data = {"a": [1], "b": ["x"]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": 1, "b": "x"}]})
    assert actual == _expected(sql, data)


def test_literal():
    sql = "SELECT 42 AS x FROM data"
    data = {"age": [1]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 1}]})
    assert actual == _expected(sql, data)


def test_arithmetic():
    sql = "SELECT age / 2 AS half FROM data"
    data = {"age": [30]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 30}]})
    assert actual == _expected(sql, data)


def test_arithmetic_precedence():
    sql = "SELECT (a + b) * c AS x FROM data"
    data = {"a": [2], "b": [3], "c": [4]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": 2, "b": 3, "c": 4}]})
    assert actual == _expected(sql, data)


def test_negative_integer_division_truncates():
    sql = "SELECT c / b AS x, c % b AS y FROM data"
    data = {"c": [-7], "b": [2]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"c": -7, "b": 2}]})
    assert actual == _expected(sql, data)


def test_mixed_int_float_division():
    sql = "SELECT a / f AS x FROM data"
    data = {"a": [7], "f": [2.5]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": 7, "f": 2.5}]})
    assert actual == _expected(sql, data)


def test_null_comparison_is_null():
    sql = "SELECT age > 100 AS gt, age > NULL AS gtnull FROM data"
    data = {"age": [30]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 30}]})
    assert actual == _expected(sql, data)
