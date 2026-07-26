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
    # multisets of repr'd rows — repr keeps value types apart (1 vs '1' vs
    # 1.0) and makes NaN compare equal to itself.
    key = lambda r: sorted((k, repr(v)) for k, v in r.items())  # noqa: E731
    assert sorted(map(key, got)) == sorted(map(key, want)), f"{got} != {want}"


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


# ------------------------------------------------------------- stretch 4:
# builtin catalogue, differential vs duckdb per the measured pins.


def test_string_builtins_differential():
    duck_check(
        "SELECT upper(s) AS u, lower(s) AS l, trim(s) AS t, ltrim(s, 'a') AS lt, "
        "rtrim(s) AS rt, substr(s, 2, 3) AS sub FROM __THIS__",
        {"s": "str?"},
        [{"s": "  aBc  "}, {"s": "abcdef"}, {"s": ""}, {"s": None}],
    )


def test_substr_edges_differential():
    duck_check(
        "SELECT substr(s, 0, 3) AS a, substr(s, -2) AS b, substr(s, -10, 8) AS c, "
        "substr(s, 1, 0) AS d, substr(s, 9) AS e FROM __THIS__",
        {"s": "str"},
        [{"s": "hello"}, {"s": "x"}],
    )


def test_concat_and_pipes_differential():
    duck_check(
        "SELECT n || '!' AS a, 'v=' || n AS b, concat(s, n, 'z') AS c, "
        "concat(s) AS d FROM __THIS__",
        {"n": "int?", "s": "str?"},
        [{"n": 1, "s": "a"}, {"n": None, "s": None}, {"n": -3, "s": ""}],
    )


def test_abs_round_differential():
    duck_check(
        "SELECT abs(n) AS an, abs(x) AS ax, round(x) AS rx, round(n) AS rn "
        "FROM __THIS__",
        {"n": "int?", "x": "float?"},
        [
            {"n": -5, "x": -2.5},
            {"n": 3, "x": 2.5},
            {"n": None, "x": None},
            {"n": 0, "x": -0.4},
        ],
    )


def test_rem_by_zero_differential():
    duck_check(
        "SELECT a % b AS r FROM __THIS__",
        {"a": "int", "b": "int"},
        [{"a": 5, "b": 0}, {"a": 5, "b": 3}, {"a": -7, "b": 2}],
    )


def test_float_rem_differential():
    duck_check(
        "SELECT x % y AS r FROM __THIS__",
        {"x": "float", "y": "float"},
        [{"x": -5.5, "y": 2.5}, {"x": 7.0, "y": 4.0}],
    )


def test_coalesce_nullif_differential():
    duck_check(
        "SELECT coalesce(n, 9) AS a, coalesce(NULL, n, 9) AS b, "
        "nullif(n, 3) AS c, nullif(s, 'x') AS d FROM __THIS__",
        {"n": "int?", "s": "str?"},
        [{"n": 3, "s": "x"}, {"n": None, "s": "y"}, {"n": 7, "s": None}],
    )


def test_nan_comparison_differential():
    # DuckDB DOUBLE order: NaN = NaN keeps the row.
    duck_check(
        "SELECT x FROM __THIS__ WHERE x = x",
        {"x": "float"},
        [{"x": float("nan")}, {"x": 1.5}],
    )


@pytest.mark.xfail(
    strict=True,
    reason="DuckDB uses utf8proc SIMPLE case maps (upper('ß')='ẞ', "
    "lower('İ')='i'); Rust std only exposes full maps — known divergence, "
    "see docs/superpowers/specs/2026-07-26-stretch4-builtin-pins.md",
)
def test_simple_case_mapping_divergence():
    duck_check(
        "SELECT upper(s) AS u, lower(s) AS l FROM __THIS__",
        {"s": "str"},
        [{"s": "ß"}, {"s": "İ"}],
    )
