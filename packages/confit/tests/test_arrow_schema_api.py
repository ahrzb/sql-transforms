"""The arrow schema surface (spec: 2026-08-13-arrow-schema-api-design.md).

Every row-table schema is a pa.Schema; rows are dict-or-object in, dict
out. Strict totality, no coercion, input range checks, nullability from
the arrow field flag. The pydantic surface is deleted.
"""

from types import SimpleNamespace

import pyarrow as pa
import pytest
from confit import DuckDBInferFn

SCHEMA = pa.schema(
    [
        pa.field("a", pa.float64()),
        pa.field("b", pa.int32()),
        pa.field("s", pa.string()),
        pa.field("k", pa.int64(), nullable=False),
    ]
)

ROW = {"a": 1.5, "b": 2, "s": "x", "k": 3}


def build(sql, schema=SCHEMA, **kw):
    return DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={}, **kw)


def test_dict_in_dict_out():
    fn = build("SELECT a + 1.0 AS o FROM __THIS__")
    out = fn.infer_rows([ROW])
    assert out == [{"o": 2.5}]
    assert isinstance(out[0], dict)


def test_object_in_dict_out():
    fn = build("SELECT k + 1 AS o FROM __THIS__")
    assert fn.infer_rows([SimpleNamespace(**ROW)]) == [{"o": 4}]


def test_narrow_width_binds_round_digits():
    # The corpus spelling pydantic could not express: round(DOUBLE, INTEGER).
    fn = build("SELECT round(a, b) AS o FROM __THIS__")
    assert fn.infer_rows([{**ROW, "a": 2.345, "b": 2}]) == [{"o": 2.35}]


def test_narrow_width_flows_to_output_schema():
    # INTEGER + INTEGER stays INTEGER in DuckDB's lattice; the declared
    # int32 must reach the binder, not collapse to int64.
    fn = build("SELECT b + b AS o FROM __THIS__")
    assert fn.output_schema.field("o").type == pa.int32()


def test_output_schema_matches_infer_arrow():
    fn = build("SELECT a * 2.0 AS d, b AS n, s AS t FROM __THIS__")
    table = fn.infer_arrow(pa.Table.from_pylist([ROW], schema=SCHEMA))
    assert fn.output_schema == table.schema


def test_missing_key_refuses_by_name():
    fn = build("SELECT a AS o FROM __THIS__")
    row = dict(ROW)
    del row["b"]
    with pytest.raises(ValueError, match="missing attribute 'b'"):
        fn.infer_rows([row])


def test_missing_attribute_refuses_by_name():
    fn = build("SELECT a AS o FROM __THIS__")
    row = dict(ROW)
    del row["s"]
    with pytest.raises(ValueError, match="missing attribute 's'"):
        fn.infer_rows([SimpleNamespace(**row)])


def test_extra_dict_keys_are_ignored():
    fn = build("SELECT a AS o FROM __THIS__")
    assert fn.infer_rows([{**ROW, "junk": object()}]) == [{"o": 1.5}]


@pytest.mark.parametrize(
    ("col", "value", "sql_type"),
    [
        ("b", True, "INTEGER"),  # bool is not an int value
        ("b", "1", "INTEGER"),
        ("b", 1.0, "INTEGER"),
        ("a", 1, "DOUBLE"),  # no int -> float coercion
        ("s", 1, "VARCHAR"),
        ("k", True, "BIGINT"),
    ],
)
def test_no_coercion(col, value, sql_type):
    fn = build("SELECT a AS o FROM __THIS__")
    with pytest.raises(ValueError, match=f"column '{col}'.*{sql_type}"):
        fn.infer_rows([{**ROW, col: value}])


def test_input_range_refuses_by_name():
    fn = build("SELECT b AS o FROM __THIS__")
    with pytest.raises(
        ValueError, match="column 'b' value 3000000000 is outside its INTEGER range"
    ):
        fn.infer_rows([{**ROW, "b": 3_000_000_000}])


def test_none_into_non_nullable_refuses():
    fn = build("SELECT k AS o FROM __THIS__")
    with pytest.raises(ValueError, match="column 'k' is not nullable"):
        fn.infer_rows([{**ROW, "k": None}])


def test_null_is_explicit_none():
    fn = build("SELECT a + 1.0 AS o FROM __THIS__")
    assert fn.infer_rows([{**ROW, "a": None}]) == [{"o": None}]


def test_bool_column_round_trip():
    schema = pa.schema([("f", pa.bool_())])
    fn = build("SELECT NOT f AS o FROM __THIS__", schema=schema)
    assert fn.infer_rows([{"f": True}, {"f": None}]) == [{"o": False}, {"o": None}]


def test_struct_row_column_serves():
    schema = pa.schema([pa.field("st", pa.struct([pa.field("x", pa.int64())]))])
    fn = build("SELECT st.x + 1 AS o FROM __THIS__", schema=schema)
    assert fn.infer_rows([{"st": {"x": 41}}]) == [{"o": 42}]
    assert fn.infer_rows([{"st": None}]) == [{"o": None}]


def test_foreign_type_unreferenced_builds_referenced_refuses():
    schema = pa.schema(
        [("a", pa.float64()), ("t", pa.timestamp("us")), ("v", pa.float32())]
    )
    fn = build("SELECT a AS o FROM __THIS__", schema=schema)
    assert fn.infer_rows([{"a": 1.0, "t": None, "v": None}]) == [{"o": 1.0}]
    with pytest.raises(ValueError, match="v"):
        build("SELECT v AS o FROM __THIS__", schema=schema)


def test_constant_static_only_query_serves_via_infer_rows():
    statics = {"s": pa.table({"v": pa.array([1, 2, 3], pa.int64())})}
    fn = DuckDBInferFn(
        "SELECT sum(v) AS o FROM s",
        row_tables={"__THIS__": SCHEMA},
        static_tables=statics,
    )
    assert fn.backend == "constant"
    assert fn.infer_rows([]) == [{"o": 6}]


def test_empty_rows_in_empty_rows_out():
    fn = build("SELECT a AS o FROM __THIS__")
    assert fn.infer_rows([]) == []


def test_infer_and_output_model_are_gone():
    fn = build("SELECT a AS o FROM __THIS__")
    assert not hasattr(fn, "infer")
    assert not hasattr(fn, "output_model")
    assert not hasattr(fn, "output")
    with pytest.raises(TypeError):
        build("SELECT a AS o FROM __THIS__", output="dict")
    with pytest.raises(TypeError):
        build("SELECT a AS o FROM __THIS__", output_model=object)


def test_non_arrow_schema_refuses_by_name():
    class Row:
        a: float

    with pytest.raises(ValueError, match="pyarrow.Schema"):
        build("SELECT a AS o FROM __THIS__", schema=Row)
