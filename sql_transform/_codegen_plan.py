"""Front-end for the codegen engine: type system and schema extraction from
Pydantic models and Arrow schemas.

Mirrors src/types.rs and src/schema.rs, but standalone -- the codegen engine
must not depend on the Rust crate. Later tasks extend this module with plan
IR, sqlglot parsing, optimization and validation (mirroring src/expr_build.rs
and src/plan.rs).
"""

from __future__ import annotations

import types as pytypes
import typing
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, create_model

INT = "int"
FLOAT = "float"
STR = "str"
BOOL = "bool"
OTHER = "other"  # unresolvable: passthrough column, union, unsupported generic


@dataclass(frozen=True)
class StructBase:
    fields: tuple  # tuple[tuple[str, FieldType], ...]; order is significant


@dataclass(frozen=True)
class ListBase:
    elem: Any  # FieldType


@dataclass(frozen=True)
class FieldType:
    base: Any
    nullable: bool


def is_container(base: Any) -> bool:
    return isinstance(base, (StructBase, ListBase))


def schema_from_pydantic(model: type[BaseModel]) -> dict:
    return dict(_pydantic_fields_ordered(model))


def _pydantic_fields_ordered(model: type[BaseModel]) -> list:
    fields = getattr(model, "model_fields", None)
    if fields is None:
        raise ValueError("Not a Pydantic v2 model class")
    return [
        (name, _annotation_to_field_type(f.annotation)) for name, f in fields.items()
    ]


def _annotation_to_field_type(annotation: Any) -> FieldType:
    origin = typing.get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            struct = StructBase(tuple(_pydantic_fields_ordered(annotation)))
            return FieldType(struct, False)
        return FieldType(_python_type_to_base(annotation), False)
    if origin is typing.Union or origin is pytypes.UnionType:
        args = typing.get_args(annotation)
        nullable = any(a is type(None) for a in args)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = _annotation_to_field_type(non_none[0])
            return FieldType(inner.base, nullable or inner.nullable)
        return FieldType(OTHER, nullable)
    if origin is list:
        args = typing.get_args(annotation)
        if len(args) == 1:
            return FieldType(ListBase(_annotation_to_field_type(args[0])), False)
    return FieldType(OTHER, False)


def _python_type_to_base(t: Any) -> Any:
    return {int: INT, float: FLOAT, str: STR, bool: BOOL}.get(t, OTHER)


def schema_from_arrow(table: Any) -> dict:
    return {f.name: _arrow_field_to_field_type(f) for f in table.schema}


def _arrow_field_to_field_type(field: Any) -> FieldType:
    return FieldType(_arrow_type_to_base(field.type), field.nullable)


def _arrow_type_to_base(t: Any) -> Any:
    if pa.types.is_struct(t):
        return StructBase(
            tuple(
                (t.field(i).name, _arrow_field_to_field_type(t.field(i)))
                for i in range(t.num_fields)
            )
        )
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return ListBase(_arrow_field_to_field_type(t.value_field))
    if pa.types.is_integer(t):
        return INT
    if pa.types.is_floating(t) or pa.types.is_decimal(t):
        return FLOAT
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return STR
    if pa.types.is_boolean(t):
        return BOOL
    return OTHER


def field_type_to_python(ft: FieldType) -> Any:
    base = ft.base
    if isinstance(base, StructBase):
        inner = {n: (field_type_to_python(f), ...) for n, f in base.fields}
        py: Any = create_model("StructModel", **inner)
    elif isinstance(base, ListBase):
        py = list[field_type_to_python(base.elem)]
    else:
        py = {INT: int, FLOAT: float, STR: str, BOOL: bool}.get(base, typing.Any)
    return py if not ft.nullable else py | None


def compatible(inferred: Any, declared: Any) -> bool:
    """Is `inferred` provably safe to store in a field declared as `declared`?
    Only rejects what can be proven wrong; Pydantic validates the rest at call
    time (mirrors types::compatible)."""
    if inferred == declared:
        return True
    if inferred == INT and declared == FLOAT:
        return True
    if inferred == OTHER:
        return True
    if isinstance(inferred, StructBase) and isinstance(declared, StructBase):
        d = dict(declared.fields)
        return len(inferred.fields) == len(declared.fields) and all(
            n in d and compatible(ft.base, d[n].base) for n, ft in inferred.fields
        )
    if isinstance(inferred, ListBase) and isinstance(declared, ListBase):
        return compatible(inferred.elem.base, declared.elem.base)
    return False
