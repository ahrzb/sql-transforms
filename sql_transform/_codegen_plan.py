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
import sqlglot
from pydantic import BaseModel, create_model
from sqlglot import expressions as exp

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


class UnsupportedInCodegen(NotImplementedError):
    """Surface the Rust InferFn covers that the codegen engine defers.

    A distinct type keeps the gap explicit at the harness boundary instead of
    letting a deferred feature silently pass as a rejection.
    """


@dataclass
class Column:
    table: str | None
    name: str


@dataclass
class Literal:
    value: Any  # native Python value; None == SQL NULL


@dataclass
class BinaryOp:
    op: str  # matches a function name in _codegen_runtime
    left: Any
    right: Any


@dataclass
class Not:
    inner: Any


@dataclass
class Func:
    name: str
    args: list


@dataclass
class Cast:
    expr: Any
    target: str


@dataclass
class TableScan:
    table: str


@dataclass
class Filter:
    input: Any
    predicate: Any


@dataclass
class CrossJoin:
    left: Any
    right: Any


@dataclass
class Join:
    left: Any
    right: Any
    on: list
    outer: bool


@dataclass
class SubqueryAlias:
    input: Any
    alias: str


@dataclass
class LookupJoin:
    input: Any
    table: str
    keys: list
    outer: bool


@dataclass
class Plan:
    projection: list
    input: Any


_BINOPS = {
    exp.Add: "add",
    exp.Sub: "sub",
    exp.Mul: "mul",
    exp.Div: "div",
    exp.Mod: "mod",
    exp.EQ: "eq",
    exp.NEQ: "neq",
    exp.LT: "lt",
    exp.GT: "gt",
    exp.LTE: "lte",
    exp.GTE: "gte",
    exp.And: "and",
    exp.Or: "or",
}

_SIMPLE_FUNCS = {
    exp.Upper: "upper",
    exp.Lower: "lower",
    exp.Abs: "abs",
    exp.Round: "round",
}
_DEFERRED_FUNCS = ("named_struct", "struct", "unnest", "make_array")

# Measured: sqlglot spreads variadic args differently PER FUNCTION, so each needs
# its own extraction. Concat  -> this=None, args all in .expressions
#                    Coalesce -> first arg in .this, rest in .expressions
#                    Nullif   -> .this + .expression
# One generic helper gets this wrong; in particular a `this not in expressions`
# de-dup guard drops an argument for COALESCE(a, a), whose sub-expressions
# compare equal -- yielding arity 1 and a silently different answer.
_CONCAT_ARGS = lambda e: list(e.expressions)  # noqa: E731
_COALESCE_ARGS = lambda e: [e.this, *e.expressions]  # noqa: E731
_NULLIF_ARGS = lambda e: [e.this, e.expression]  # noqa: E731
_VARIADIC_FUNCS = {
    exp.Concat: ("concat", _CONCAT_ARGS),
    exp.Coalesce: ("coalesce", _COALESCE_ARGS),
    exp.Nullif: ("nullif", _NULLIF_ARGS),
}


def _convert_expr(e: exp.Expression) -> Any:
    if isinstance(e, (exp.Paren, exp.Alias)):
        return _convert_expr(e.this)
    if isinstance(e, exp.Dot):
        raise UnsupportedInCodegen(
            "struct field access is not supported in codegen yet"
        )
    if isinstance(e, exp.Column):
        # `s.a.b` parses as a 3-part Column carrying `db` (NOT exp.Dot). Reading
        # only .table/.name would silently misread it as Column('a', 'b') and drop
        # `s` -- a wrong answer rather than an error. It is struct field access,
        # which is deferred.
        if e.args.get("db") or e.args.get("catalog"):
            raise UnsupportedInCodegen(
                "struct field access is not supported in codegen yet"
            )
        return Column(table=e.table or None, name=e.name)
    if isinstance(e, exp.Null):
        return Literal(None)
    if isinstance(e, exp.Boolean):
        return Literal(bool(e.this))
    if isinstance(e, exp.Literal):
        return Literal(_convert_literal(e))
    if isinstance(e, exp.Neg):
        inner = _convert_expr(e.this)
        if isinstance(inner, Literal) and type(inner.value) in (int, float):
            return Literal(-inner.value)
        raise ValueError(f"Unsupported expression: {e.sql()}")
    if isinstance(e, exp.Not):
        return Not(_convert_expr(e.this))
    for cls, op in _BINOPS.items():
        if isinstance(e, cls):
            return BinaryOp(op, _convert_expr(e.this), _convert_expr(e.expression))
    if isinstance(e, exp.Cast):
        return Cast(_convert_expr(e.this), _cast_target(e.to.sql()))
    if isinstance(e, (exp.Struct, exp.Array)):
        raise UnsupportedInCodegen(
            "struct/list construction is not supported in codegen yet"
        )
    if isinstance(e, exp.Unnest):
        # Measured: UNNEST(...) parses as its own exp.Unnest class, not
        # exp.Anonymous -- the _DEFERRED_FUNCS name-matching branch below never
        # sees it. Without this check it falls through to the generic
        # ValueError instead of the UnsupportedInCodegen the harness expects
        # for deferred container ops.
        raise UnsupportedInCodegen("unnest() is not supported in codegen yet")
    if isinstance(e, exp.Trim):
        if e.args.get("position") or e.expression:
            raise ValueError("Only plain TRIM(expr) is supported")
        return Func("trim", [_convert_expr(e.this)])
    if isinstance(e, exp.Substring):
        args = [_convert_expr(e.this)]
        start = e.args.get("start")
        args.append(_convert_expr(start) if start is not None else Literal(1))
        length = e.args.get("length")
        if length is not None:
            args.append(_convert_expr(length))
        return Func("substr", args)
    for cls, name in _SIMPLE_FUNCS.items():
        if isinstance(e, cls):
            return Func(name, [_convert_expr(e.this)])
    for cls, (name, extract) in _VARIADIC_FUNCS.items():
        if isinstance(e, cls):
            return Func(name, [_convert_expr(a) for a in extract(e)])
    if isinstance(e, exp.Anonymous):
        name = e.name.lower()
        if name in _DEFERRED_FUNCS:
            raise UnsupportedInCodegen(f"{name}() is not supported in codegen yet")
        return Func(name, [_convert_expr(a) for a in e.expressions])
    raise ValueError(f"Unsupported expression: {e.sql()}")


def _convert_literal(e: exp.Literal) -> Any:
    if e.is_string:
        return e.this
    text = e.this
    return float(text) if "." in text or "e" in text.lower() else int(text)


def _cast_target(name: str) -> str:
    name = name.upper()
    if name.startswith(("VARCHAR", "TEXT", "STRING", "CHAR")):
        return STR
    if name.startswith(("BIGINT", "INT")):
        return INT
    if name.startswith(("DOUBLE", "FLOAT", "REAL", "DECIMAL")):
        return FLOAT
    if name.startswith("BOOL"):
        return BOOL
    raise ValueError(f"Unsupported CAST target type: {name}")


def build_plan(sql: str) -> Plan:
    tree = sqlglot.parse_one(sql)
    if not isinstance(tree, exp.Select):
        raise ValueError("Only SELECT queries are supported")
    node = _build_from(tree)
    where = tree.args.get("where")
    if where is not None:
        node = Filter(node, _convert_expr(where.this))
    return Plan(_build_projection(tree.expressions), node)


def _build_from(tree: exp.Select) -> Any:
    # sqlglot 30 renamed this arg "from" -> "from_". Reading "from" silently
    # returns None, which would reject every query as missing a FROM clause.
    from_ = tree.args.get("from_")
    if from_ is None:
        raise ValueError("FROM clause is required")
    seen: set = set()
    node = _table_factor(from_.this, seen)
    for join in tree.args.get("joins") or []:
        node = _build_join(node, join, seen)
    return node


def _build_join(left: Any, join: exp.Join, seen: set) -> Any:
    right = _table_factor(join.this, seen)
    kind = (join.args.get("kind") or "").upper()
    side = (join.args.get("side") or "").upper()
    on = join.args.get("on")
    if kind == "CROSS" or (on is None and not side and not kind):
        return CrossJoin(left, right)
    if on is None:
        raise ValueError("JOIN requires an ON condition")
    if side not in ("", "LEFT"):
        raise ValueError(
            f"Unsupported JOIN type: {side} {kind} — only inner JOIN ... ON, "
            "LEFT JOIN ... ON and CROSS JOIN are supported"
        )
    return Join(left, right, _equality_keys(on), side == "LEFT")


def _table_factor(factor: exp.Expression, seen: set) -> Any:
    if not isinstance(factor, exp.Table):
        raise ValueError("Unsupported FROM clause")
    name = factor.name
    alias = factor.alias or None
    # Track the EFFECTIVE name: a collision would silently overwrite one side's
    # data when rows merge (plan.rs build_table_factor).
    effective = alias or name
    if effective in seen:
        raise ValueError(
            f"table '{effective}' is referenced more than once in FROM/JOIN — "
            "self-joins and alias collisions are not supported"
        )
    seen.add(effective)
    scan = TableScan(name)
    return SubqueryAlias(scan, alias) if alias else scan


def _equality_keys(e: exp.Expression) -> list:
    if isinstance(e, exp.Paren):
        return _equality_keys(e.this)
    if isinstance(e, exp.And):
        return _equality_keys(e.this) + _equality_keys(e.expression)
    if isinstance(e, exp.EQ):
        return [(_convert_expr(e.this), _convert_expr(e.expression))]
    raise ValueError(
        "JOIN ON condition must be an equality, or an AND of equalities, "
        "between columns"
    )


def _build_projection(items: list) -> list:
    out = []
    for item in items:
        if isinstance(item, exp.Alias):
            out.append((item.alias, _convert_expr(item.this)))
        elif isinstance(item, exp.Column):
            out.append((item.name, _convert_expr(item)))
        elif isinstance(item, exp.Star):
            raise ValueError("Unsupported SELECT item: *")
        else:
            raise ValueError("Expression in SELECT list needs an alias (AS name)")
    return out
