"""Unit tests for the codegen front-end (types, parse, optimize, validate)."""

import typing

import pyarrow as pa
from pydantic import BaseModel

from sql_transform import _codegen_plan as cp


def test_schema_from_pydantic_reads_types_and_nullability():
    class Row(BaseModel):
        a: int
        b: float | None
        c: str
        d: bool

    schema = cp.schema_from_pydantic(Row)
    assert schema["a"] == cp.FieldType(cp.INT, False)
    assert schema["b"] == cp.FieldType(cp.FLOAT, True)
    assert schema["c"] == cp.FieldType(cp.STR, False)
    assert schema["d"] == cp.FieldType(cp.BOOL, False)


def test_schema_from_pydantic_optional_and_any():
    class Row(BaseModel):
        a: int | None
        b: typing.Any

    schema = cp.schema_from_pydantic(Row)
    assert schema["a"] == cp.FieldType(cp.INT, True)
    assert schema["b"].base == cp.OTHER


def test_schema_from_pydantic_nested_model_is_a_struct():
    class Inner(BaseModel):
        x: int

    class Row(BaseModel):
        s: Inner

    schema = cp.schema_from_pydantic(Row)
    assert schema["s"].base == cp.StructBase((("x", cp.FieldType(cp.INT, False)),))
    assert cp.is_container(schema["s"].base)


def test_schema_from_arrow_reads_types_and_nullability():
    table = pa.table(
        {"a": pa.array([1], type=pa.int64()), "b": pa.array([1.0], type=pa.float64())},
        schema=pa.schema(
            [pa.field("a", pa.int64(), nullable=False), pa.field("b", pa.float64())]
        ),
    )
    schema = cp.schema_from_arrow(table)
    assert schema["a"] == cp.FieldType(cp.INT, False)
    assert schema["b"] == cp.FieldType(cp.FLOAT, True)


def test_field_type_to_python_round_trips():
    assert cp.field_type_to_python(cp.FieldType(cp.INT, False)) is int
    assert cp.field_type_to_python(cp.FieldType(cp.INT, True)) == (int | None)
    assert cp.field_type_to_python(cp.FieldType(cp.OTHER, False)) is typing.Any


def test_compatible_allows_int_into_float_and_unknown_into_anything():
    assert cp.compatible(cp.INT, cp.FLOAT)
    assert cp.compatible(cp.OTHER, cp.STR)
    assert cp.compatible(cp.INT, cp.INT)
    assert not cp.compatible(cp.STR, cp.INT)
    assert not cp.compatible(cp.FLOAT, cp.INT)
    assert cp.is_container(cp.INT) is False


def test_compatible_struct_and_list():
    x_int = ("x", cp.FieldType(cp.INT, False))
    y_str = ("y", cp.FieldType(cp.STR, False))

    # reordered same-name-same-type structs are compatible (order-independent).
    inferred = cp.StructBase((x_int, y_str))
    declared_reordered = cp.StructBase((y_str, x_int))
    assert cp.compatible(inferred, declared_reordered)
    assert cp.is_container(inferred)

    # widening a field INT->FLOAT inside a struct is compatible.
    declared_widened = cp.StructBase((("x", cp.FieldType(cp.FLOAT, False)), y_str))
    assert cp.compatible(inferred, declared_widened)

    # different field names -> not compatible.
    declared_diff_name = cp.StructBase(
        (("x", cp.FieldType(cp.INT, False)), ("z", cp.FieldType(cp.STR, False)))
    )
    assert not cp.compatible(inferred, declared_diff_name)

    # different field count -> not compatible.
    declared_diff_count = cp.StructBase((x_int,))
    assert not cp.compatible(inferred, declared_diff_count)

    # incompatible field type -> not compatible.
    declared_diff_type = cp.StructBase((("x", cp.FieldType(cp.STR, False)), y_str))
    assert not cp.compatible(inferred, declared_diff_type)

    # list element base compatibility mirrors scalar rules.
    list_int = cp.ListBase(cp.FieldType(cp.INT, False))
    list_float = cp.ListBase(cp.FieldType(cp.FLOAT, False))
    list_str = cp.ListBase(cp.FieldType(cp.STR, False))
    assert cp.compatible(list_int, list_float)
    assert not cp.compatible(list_str, list_int)
    assert cp.is_container(list_int)

    # struct vs list -> not compatible.
    assert not cp.compatible(inferred, list_int)
