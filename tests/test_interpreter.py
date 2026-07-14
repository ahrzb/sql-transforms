"""Tests for the Rust SQL interpreter (sql_transform._interpreter).

Row tables are declared as Pydantic v2 model classes at InferFn construction
time; row inputs to .infer() are instances of those models. Every
behavioral test that involves computed values compares InferFn output
against real DataFusion batch output for the same SQL + data.
"""

import datafusion
import pyarrow as pa
import pytest
from pydantic import BaseModel

from sql_transform._interpreter import InferFn


def _expected(sql: str, data: dict) -> list[dict]:
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name="data")
    return ctx.sql(sql).collect()[0].to_pylist()


class Data(BaseModel):
    age: int
    name: str | None = None


def test_module_imports_and_constructs():
    fn = InferFn("SELECT age FROM data", row_tables={"data": Data}, static_tables={})
    assert fn is not None


def test_column_pass_through():
    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=30)]})
    assert actual == _expected(sql, {"age": [30]})


def test_arithmetic_and_where():
    sql = "SELECT age, age * 2 AS doubled FROM data WHERE age > 18"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=15), Data(age=25), Data(age=40)]})
    assert actual == _expected(sql, {"age": [15, 25, 40]})


def test_builtin_function_and_cast():
    sql = "SELECT UPPER(name) AS n, CAST(age AS VARCHAR) AS s FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=30, name="alice")]})
    assert actual == _expected(sql, {"age": [30], "name": ["alice"]})


class A(BaseModel):
    id: int
    x: int


class B(BaseModel):
    id: int
    y: int


def test_cross_join():
    sql = "SELECT a.x, b.y FROM a, b"
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"id": [1], "x": [10]}, name="a")
    ctx.from_pydict({"id": [1], "y": [20]}, name="b")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables={"a": A, "b": B}, static_tables={})
    actual = fn.infer({"a": [A(id=1, x=10)], "b": [B(id=1, y=20)]})
    assert actual == expected


def test_inner_join():
    sql = "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id"
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"id": [1, 2], "x": [10, 20]}, name="a")
    ctx.from_pydict({"id": [1, 2], "y": [100, 200]}, name="b")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables={"a": A, "b": B}, static_tables={})
    actual = fn.infer(
        {
            "a": [A(id=1, x=10), A(id=2, x=20)],
            "b": [B(id=1, y=100), B(id=2, y=200)],
        }
    )
    assert actual == expected


def test_aliased_row_table():
    sql = "SELECT d.age FROM data AS d WHERE d.age > 18"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=15), Data(age=25)]})
    assert actual == _expected(sql, {"age": [15, 25]})


def test_join_row_and_static_table():
    ref_table = pa.table({"id": [1, 2], "y": [10, 20]})
    sql = "SELECT data.x, ref.y FROM data JOIN ref ON data.id = ref.id"

    class RowWithId(BaseModel):
        id: int
        x: int

    ctx = datafusion.SessionContext()
    ctx.from_pydict({"id": [1, 2], "x": [5, 6]}, name="data")
    ctx.from_arrow(ref_table, name="ref")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables={"data": RowWithId}, static_tables={"ref": ref_table})
    actual = fn.infer({"data": [RowWithId(id=1, x=5), RowWithId(id=2, x=6)]})
    assert actual == expected


def test_error_unknown_row_column():
    sql = "SELECT nonexistent FROM data"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables={"data": Data}, static_tables={})


def test_error_unknown_static_column():
    ref_table = pa.table({"id": [1], "y": [10]})
    sql = "SELECT data.age, ref.nonexistent FROM data JOIN ref ON data.age = ref.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables={"data": Data}, static_tables={"ref": ref_table})


def test_error_self_join_still_rejected():
    sql = "SELECT a.x FROM a JOIN a ON a.id = a.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables={"a": A}, static_tables={})
