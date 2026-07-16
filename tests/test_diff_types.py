import pyarrow as pa
import pytest
from differential import check, rows

from sql_transform import SQLTransform


def test_struct_construct():
    check(
        "SELECT named_struct('x', a, 'y', b) AS s FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": 1, "b": 2}])},
    )


def test_list_construct():
    check(
        "SELECT [a, b, a] AS l FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": 1, "b": 2}])},
    )


def test_struct_input_roundtrip():
    check(
        "SELECT s FROM t",
        {"t": rows({"s": "struct{x:int,y:int}"}, [{"s": {"x": 1, "y": 2}}])},
    )


def test_list_input_roundtrip():
    check(
        "SELECT l FROM t",
        {"t": rows({"l": "list[int]"}, [{"l": [1, 2, 3]}])},
    )


def test_malformed_struct_input_raises():
    # A scalar where a struct is declared must error, not silently marshal
    # into an all-null struct. Direct infer() test: the differential check()
    # harness validates rows via model(**r) before they reach the Rust side,
    # so it can't exercise this path.
    schema = pa.schema([pa.field("s", pa.struct([("x", pa.int64()), ("y", pa.int64())]))])
    table = pa.Table.from_pylist([{"s": {"x": 1, "y": 2}}], schema=schema)
    t = SQLTransform("SELECT s FROM __THIS__").fit(table)
    with pytest.raises(ValueError):
        t.infer({"s": 5})


def test_deep_nesting_roundtrip():
    check(
        "SELECT s FROM t",
        {
            "t": rows(
                {"s": "struct{a:int,inner:list[int]}"},
                [{"s": {"a": 1, "inner": [1, 2, 3]}}],
            )
        },
    )


def test_struct_field_access():
    check("SELECT s.x AS fx FROM t",
          {"t": rows({"s": "struct{x:int,y:int}"},
                     [{"s": {"x": 5, "y": 9}}])})
