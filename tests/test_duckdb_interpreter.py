"""DuckDBInferFn vs duckdb-python: the specializer's differential oracle.

`duck_check` runs the same SQL on the same data through both engines and
asserts the outputs agree row-for-row. Stretch-3 surface: the projection /
WHERE spine plus equi-joins to static tables (INNER and LEFT), which lower
to map probes.
"""

from __future__ import annotations

from typing import Any

import duckdb
import pyarrow as pa
import pytest
from pydantic import create_model

from sql_transform._interpreter import DuckDBInferFn

_PY = {"int": int, "float": float, "str": str, "bool": bool}
_ARROW = {
    "int": pa.int64(),
    "float": pa.float64(),
    "str": pa.string(),
    "bool": pa.bool_(),
}


def _row_model(schema: dict[str, str]):
    fields: dict[str, Any] = {}
    for name, spec in schema.items():
        if spec.endswith("?"):
            fields[name] = (_PY[spec[:-1]] | None, None)
        else:
            fields[name] = (_PY[spec], ...)
    return create_model("Row", **fields)


def static(schema: dict[str, str], rows: list[dict[str, Any]]) -> pa.Table:
    arrow = pa.schema(
        pa.field(n, _ARROW[s.rstrip("?")], nullable=s.endswith("?"))
        for n, s in schema.items()
    )
    return pa.Table.from_pylist(rows, schema=arrow)


def duck_check(
    sql: str,
    row_schema: dict[str, str],
    row_rows: list[dict[str, Any]],
    statics: dict[str, pa.Table] | None = None,
) -> None:
    statics = statics or {}
    model = _row_model(row_schema)
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": model}, static_tables=statics)
    inputs = [model(**r) for r in row_rows]
    got = [r.model_dump() for r in fn.infer({"__THIS__": inputs})]

    con = duckdb.connect()
    for name, table in statics.items():
        con.register(name, table)
    con.register("__THIS__", static(row_schema, row_rows))
    want = con.execute(sql).to_arrow_table().to_pylist()

    # Row order is not part of the contract (a join may reorder); compare as
    # multisets. NaN-free data only — keep NaN semantics in the Rust tests.
    key = lambda r: sorted((k, str(v)) for k, v in r.items())  # noqa: E731
    assert sorted(got, key=key) == sorted(want, key=key), f"{got} != {want}"


DIM = static(
    {"id": "int", "name": "str"},
    [{"id": 1, "name": "one"}, {"id": 3, "name": "three"}],
)


def test_projection_and_where_differential():
    duck_check(
        "SELECT a + 1 AS x, a / 2 AS h FROM __THIS__ WHERE a > 0",
        {"a": "int"},
        [{"a": 4}, {"a": -1}, {"a": 7}],
    )


def test_inner_join_hits_and_misses():
    duck_check(
        "SELECT k, name FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int"},
        [{"k": 1}, {"k": 2}, {"k": 3}],
        {"dim": DIM},
    )


def test_left_join_miss_and_null_key():
    duck_check(
        "SELECT k, name FROM __THIS__ LEFT JOIN dim ON k = dim.id",
        {"k": "int?"},
        [{"k": 1}, {"k": 2}, {"k": None}],
        {"dim": DIM},
    )


def test_inner_join_null_key_drops():
    duck_check(
        "SELECT name FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int?"},
        [{"k": 1}, {"k": None}],
        {"dim": DIM},
    )


def test_join_key_promotion_int_row_against_float_col():
    duck_check(
        "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int"},
        [{"k": 1}, {"k": 2}],
        {"dim": static({"id": "float", "v": "int"}, [{"id": 1.0, "v": 10}])},
    )


def test_join_key_expression_and_where_interaction():
    duck_check(
        "SELECT k, name FROM __THIS__ JOIN dim ON k + 1 = dim.id WHERE k >= 0",
        {"k": "int"},
        [{"k": 0}, {"k": 2}, {"k": -5}],
        {"dim": DIM},
    )


def test_duplicate_build_keys_error():
    dup = static({"id": "int", "v": "int"}, [{"id": 1, "v": 10}, {"id": 1, "v": 11}])
    with pytest.raises(ValueError, match="duplicate map key"):
        DuckDBInferFn(
            "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
            row_tables={"__THIS__": _row_model({"k": "int"})},
            static_tables={"dim": dup},
        )


def test_null_in_value_column_errors():
    holed = static({"id": "int", "v": "int?"}, [{"id": 1, "v": None}])
    with pytest.raises(ValueError, match="NULL in value column"):
        DuckDBInferFn(
            "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
            row_tables={"__THIS__": _row_model({"k": "int"})},
            static_tables={"dim": holed},
        )


def test_null_key_build_rows_are_dropped():
    # A NULL build key never equi-matches, so the row is dropped rather than
    # rejected — probing k=1 still hits the valid entry.
    holed = static(
        {"id": "int?", "v": "int"}, [{"id": 1, "v": 10}, {"id": None, "v": 99}]
    )
    duck_check(
        "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int"},
        [{"k": 1}],
        {"dim": holed},
    )


def test_unsupported_is_a_clean_value_error():
    with pytest.raises(ValueError, match="unsupported.*GROUP BY"):
        DuckDBInferFn(
            "SELECT a FROM __THIS__ GROUP BY a",
            row_tables={"__THIS__": _row_model({"a": "int"})},
            static_tables={},
        )


def test_bad_sql_is_a_build_error():
    with pytest.raises(ValueError, match="parse error"):
        DuckDBInferFn(
            "SELECT FROM",
            row_tables={"__THIS__": _row_model({"a": "int"})},
            static_tables={},
        )


def test_unknown_infer_table_is_rejected():
    model = _row_model({"a": "int"})
    fn = DuckDBInferFn(
        "SELECT a FROM __THIS__", row_tables={"__THIS__": model}, static_tables={}
    )
    with pytest.raises(ValueError, match="unknown table"):
        fn.infer({"wrong": [model(a=1)]})


def test_output_model_is_synthesized():
    fn = DuckDBInferFn(
        "SELECT a + 1 AS x, b AS y FROM __THIS__",
        row_tables={"__THIS__": _row_model({"a": "int", "b": "float?"})},
        static_tables={},
    )
    fields = fn.output_model.model_fields
    assert list(fields) == ["x", "y"]
    assert fields["x"].annotation is int
    assert fields["y"].annotation == float | None
