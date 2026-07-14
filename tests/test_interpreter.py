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
