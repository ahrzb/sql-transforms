"""Synthesize a Pydantic model for SQLTransform's __THIS__ table."""

from __future__ import annotations

import typing

import pyarrow as pa
from pydantic import BaseModel, create_model


def synthesize_this_model(schema: pa.Schema) -> type[BaseModel]:
    """Build a Pydantic model matching an Arrow schema's columns, types, and
    nullability — used as __THIS__'s row schema when the caller doesn't
    supply their own `this_model`."""
    fields = {field.name: (_arrow_field_to_python(field), ...) for field in schema}
    return create_model("ThisRow", **fields)


def _arrow_field_to_python(field: pa.Field) -> object:
    base = _arrow_type_to_python(field.type)
    return base | None if base is not typing.Any and field.nullable else base


def _arrow_type_to_python(arrow_type: pa.DataType) -> object:
    if pa.types.is_integer(arrow_type):
        return int
    if pa.types.is_floating(arrow_type):
        return float
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return str
    if pa.types.is_boolean(arrow_type):
        return bool
    if pa.types.is_struct(arrow_type):
        fields = {
            arrow_type.field(i).name: (_arrow_field_to_python(arrow_type.field(i)), ...)
            for i in range(arrow_type.num_fields)
        }
        return create_model("StructField", **fields)
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return list[_arrow_field_to_python(arrow_type.value_field)]
    return typing.Any
