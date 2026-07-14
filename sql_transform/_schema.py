"""Synthesize Pydantic models for SQLTransform's __THIS__ and __STATE__ tables."""

from __future__ import annotations

import typing

import pyarrow as pa
from pydantic import BaseModel, create_model


def synthesize_this_model(schema: pa.Schema) -> type[BaseModel]:
    """Build a Pydantic model matching an Arrow schema's columns, types, and
    nullability — used as __THIS__'s row schema when the caller doesn't
    supply their own `this_model`."""
    fields: dict[str, tuple[object, object]] = {}
    for field in schema:
        base = _arrow_type_to_python(field.type)
        py_type: object = base
        if base is not typing.Any and field.nullable:
            py_type = base | None
        fields[field.name] = (py_type, ...)
    return create_model("ThisRow", **fields)


def synthesize_state_model(state: dict[str, float]) -> type[BaseModel]:
    """Build a Pydantic model with one float field per state key — used as
    __STATE__'s row schema. Field names must have no leading underscore:
    Pydantic v2 treats those as private attributes, excluded from
    model_fields and unsettable via the constructor."""
    fields = dict.fromkeys(state, (float, ...))
    return create_model("StateModel", **fields)


def _arrow_type_to_python(arrow_type: pa.DataType) -> object:
    if pa.types.is_integer(arrow_type):
        return int
    if pa.types.is_floating(arrow_type):
        return float
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return str
    if pa.types.is_boolean(arrow_type):
        return bool
    return typing.Any
