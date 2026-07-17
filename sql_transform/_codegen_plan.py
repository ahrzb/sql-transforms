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
        elif isinstance(item, exp.Unnest):
            # A bare `unnest(...)` (no AS) never reaches the exp.Alias branch
            # above, so without this it falls to the generic "needs an alias"
            # ValueError below instead of the UnsupportedInCodegen the harness
            # expects for deferred container ops.
            _convert_expr(item)
        else:
            raise ValueError("Expression in SELECT list needs an alias (AS name)")
    return out


@dataclass
class LookupSpec:
    static_table: str
    key_columns: list


def optimize(plan: Plan, static_tables: set) -> tuple:
    """Rewrite every Join with exactly one static side into a LookupJoin
    (mirrors plan::optimize)."""
    specs: list = []
    return Plan(plan.projection, _optimize_rel(plan.input, static_tables, specs)), specs


def _optimize_rel(node: Any, static_tables: set, specs: list) -> Any:
    if isinstance(node, Join):
        left = _optimize_rel(node.left, static_tables, specs)
        right = _optimize_rel(node.right, static_tables, specs)
        left_name, right_name = scan_name(left), scan_name(right)
        left_static = left_name if left_name in static_tables else None
        right_static = right_name if right_name in static_tables else None
        if left_static and right_static:
            raise ValueError("Joining two static tables together is not supported")
        if right_static or left_static:
            table = right_static or left_static
            other = left if right_static else right
            keys, key_columns = _split_keys(node.on, table)
            specs.append(LookupSpec(table, key_columns))
            return LookupJoin(other, table, keys, node.outer)
        if node.outer:
            raise ValueError(
                "LEFT JOIN is only supported against a static lookup table"
            )
        return Join(left, right, node.on, node.outer)
    if isinstance(node, CrossJoin):
        return CrossJoin(
            _optimize_rel(node.left, static_tables, specs),
            _optimize_rel(node.right, static_tables, specs),
        )
    if isinstance(node, Filter):
        return Filter(_optimize_rel(node.input, static_tables, specs), node.predicate)
    if isinstance(node, SubqueryAlias):
        return SubqueryAlias(
            _optimize_rel(node.input, static_tables, specs), node.alias
        )
    return node


def scan_name(node: Any) -> str | None:
    """The real table name a scan (possibly aliased) reads from."""
    if isinstance(node, TableScan):
        return node.table
    if isinstance(node, SubqueryAlias):
        return scan_name(node.input)
    return None


def _split_keys(on: list, static_table: str) -> tuple:
    """Split each ON equality into (row-side expression, static key column).
    The static side is identified per-pair by qualifier, since `a = b` vs
    `b = a` is independent of which side is structurally left/right."""
    row_keys, static_cols = [], []
    for left, right in on:
        lq = left.table if isinstance(left, Column) else None
        rq = right.table if isinstance(right, Column) else None
        if lq == static_table:
            static_expr, row_expr = left, right
        elif rq == static_table:
            static_expr, row_expr = right, left
        else:
            raise ValueError(
                f"JOIN ON keys against static table '{static_table}' must reference "
                f"the static table's columns by name (e.g. {static_table}.col)"
            )
        if not isinstance(static_expr, Column):
            raise ValueError(
                f"JOIN ON keys against static table '{static_table}' must be "
                "plain columns"
            )
        static_cols.append(static_expr.name)
        row_keys.append(row_expr)
    return row_keys, static_cols


@dataclass
class ColumnValidation:
    row_table_columns: dict
    effective_schemas: dict


def validate_columns(
    plan: Plan, row_table_names: set, row_schemas: dict, static_schemas: dict
) -> ColumnValidation:
    """Validate every column reference against the resolved table schemas,
    rewrite unqualified refs to their effective table, and collect (per row
    table's REAL name) the columns the query actually reads."""
    resolved: dict = {}
    nullable_tables: set = set()
    _resolve_tables(plan.input, row_table_names, False, resolved, nullable_tables)

    effective_schemas: dict = {}
    for effective, (real, is_row) in resolved.items():
        schema = (row_schemas if is_row else static_schemas).get(real)
        if schema is None:
            continue
        if effective in nullable_tables:
            # An unmatched outer row makes every column on that side NULL, so
            # the synthesized output type must be nullable even when the source
            # declares otherwise.
            schema = {k: FieldType(v.base, True) for k, v in schema.items()}
        effective_schemas[effective] = schema

    used: dict = {}
    for _, e in plan.projection:
        _validate_expr(e, resolved, row_schemas, static_schemas, used)
    _validate_rel(plan.input, resolved, row_schemas, static_schemas, used)
    return ColumnValidation({k: sorted(v) for k, v in used.items()}, effective_schemas)


def _resolve_tables(
    node: Any, row_table_names: set, nullable: bool, out: dict, nullable_out: set
) -> None:
    if isinstance(node, TableScan):
        out[node.table] = (node.table, node.table in row_table_names)
        if nullable:
            nullable_out.add(node.table)
    elif isinstance(node, SubqueryAlias):
        real = scan_name(node.input)
        if real is not None:
            out[node.alias] = (real, real in row_table_names)
            if nullable:
                nullable_out.add(node.alias)
    elif isinstance(node, Filter):
        _resolve_tables(node.input, row_table_names, nullable, out, nullable_out)
    elif isinstance(node, CrossJoin):
        _resolve_tables(node.left, row_table_names, nullable, out, nullable_out)
        _resolve_tables(node.right, row_table_names, nullable, out, nullable_out)
    elif isinstance(node, Join):
        _resolve_tables(node.left, row_table_names, nullable, out, nullable_out)
        _resolve_tables(
            node.right, row_table_names, nullable or node.outer, out, nullable_out
        )
    elif isinstance(node, LookupJoin):
        _resolve_tables(node.input, row_table_names, nullable, out, nullable_out)
        out[node.table] = (node.table, False)
        if nullable or node.outer:
            nullable_out.add(node.table)


def _validate_rel(node: Any, resolved, row_schemas, static_schemas, used) -> None:
    if isinstance(node, Filter):
        _validate_expr(node.predicate, resolved, row_schemas, static_schemas, used)
        _validate_rel(node.input, resolved, row_schemas, static_schemas, used)
    elif isinstance(node, CrossJoin):
        _validate_rel(node.left, resolved, row_schemas, static_schemas, used)
        _validate_rel(node.right, resolved, row_schemas, static_schemas, used)
    elif isinstance(node, Join):
        for left, right in node.on:
            _validate_expr(left, resolved, row_schemas, static_schemas, used)
            _validate_expr(right, resolved, row_schemas, static_schemas, used)
        _validate_rel(node.left, resolved, row_schemas, static_schemas, used)
        _validate_rel(node.right, resolved, row_schemas, static_schemas, used)
    elif isinstance(node, SubqueryAlias):
        _validate_rel(node.input, resolved, row_schemas, static_schemas, used)
    elif isinstance(node, LookupJoin):
        for k in node.keys:
            _validate_expr(k, resolved, row_schemas, static_schemas, used)
        _validate_rel(node.input, resolved, row_schemas, static_schemas, used)


def _validate_expr(e: Any, resolved, row_schemas, static_schemas, used) -> None:
    if isinstance(e, Column):
        if e.table is not None:
            entry = resolved.get(e.table)
            if entry is None:
                # `s.x` (2-part struct field access, e.g. s is struct{x,y})
                # parses identically to a table-qualified column -- with no
                # `db`/`catalog` to flag it, unlike the 3-part `t.s.x` case in
                # _convert_expr. Without this check it falls to the generic
                # "unknown table" ValueError instead of the UnsupportedInCodegen
                # the harness expects for deferred container ops.
                for schema in (*row_schemas.values(), *static_schemas.values()):
                    ft = (schema or {}).get(e.table)
                    if ft is not None and is_container(ft.base):
                        raise UnsupportedInCodegen(
                            "struct field access is not supported in codegen yet"
                        )
                raise ValueError(f"Unknown table: {e.table}")
            real, is_row = entry
            schema = (row_schemas if is_row else static_schemas).get(real)
            if schema is None:
                raise ValueError(f"Unknown table: {real}")
            if e.name not in schema:
                raise ValueError(f"Unknown column: {real}.{e.name}")
            if is_row:
                used.setdefault(real, set()).add(e.name)
            return
        matches = [
            (effective, real, is_row)
            for effective, (real, is_row) in resolved.items()
            if e.name in ((row_schemas if is_row else static_schemas).get(real) or {})
        ]
        if not matches:
            raise ValueError(f"Unknown column: {e.name}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous column reference: {e.name}")
        effective, real, is_row = matches[0]
        e.table = effective  # codegen emits a direct subscript off this
        if is_row:
            used.setdefault(real, set()).add(e.name)
    elif isinstance(e, BinaryOp):
        _validate_expr(e.left, resolved, row_schemas, static_schemas, used)
        _validate_expr(e.right, resolved, row_schemas, static_schemas, used)
    elif isinstance(e, Not):
        _validate_expr(e.inner, resolved, row_schemas, static_schemas, used)
    elif isinstance(e, Cast):
        _validate_expr(e.expr, resolved, row_schemas, static_schemas, used)
    elif isinstance(e, Func):
        for a in e.args:
            _validate_expr(a, resolved, row_schemas, static_schemas, used)


_STR_FUNCS = frozenset({"upper", "lower", "trim", "substr", "substring"})


def infer_type(e: Any, schemas: dict) -> FieldType:
    """Statically infer a projection's FieldType, mirroring types::infer_type.
    Sound but not tight on nullability: nullable means "cannot prove non-NULL"."""
    if isinstance(e, Column):
        return _resolve_column_type(e.table, e.name, schemas)
    if isinstance(e, Literal):
        return _literal_type(e.value)
    if isinstance(e, BinaryOp):
        left, right = infer_type(e.left, schemas), infer_type(e.right, schemas)
        nullable = left.nullable or right.nullable
        if e.op in ("add", "sub", "mul", "div", "mod"):
            base = INT if left.base == INT and right.base == INT else FLOAT
            return FieldType(base, nullable)
        if is_container(left.base) or is_container(right.base):
            # Comparing/combining structs or lists needs deep equality, which
            # the runtime doesn't implement -- without this it silently reaches
            # the scalar-only arithmetic comparison path and crashes instead of
            # being reported as a deferred surface.
            raise UnsupportedInCodegen(
                "struct/list comparison is not supported in codegen yet"
            )
        return FieldType(BOOL, nullable)
    if isinstance(e, Not):
        return FieldType(BOOL, infer_type(e.inner, schemas).nullable)
    if isinstance(e, Cast):
        return FieldType(e.target, infer_type(e.expr, schemas).nullable)
    if isinstance(e, Func):
        return _function_type(e.name, [infer_type(a, schemas) for a in e.args])
    raise UnsupportedInCodegen(f"cannot infer the type of {type(e).__name__}")


def _resolve_column_type(table: str | None, name: str, schemas: dict) -> FieldType:
    if table is not None:
        schema = schemas.get(table)
        if schema is None or name not in schema:
            raise ValueError(f"Unknown column: {table}.{name}")
        return schema[name]
    found = None
    for schema in schemas.values():
        if name in schema:
            if found is not None:
                raise ValueError(f"Ambiguous column reference: {name}")
            found = schema[name]
    if found is None:
        raise ValueError(f"Unknown column: {name}")
    return found


def _literal_type(v: Any) -> FieldType:
    if v is None:
        return FieldType(OTHER, True)
    t = type(v)
    if t is bool:
        return FieldType(BOOL, False)
    if t is int:
        return FieldType(INT, False)
    if t is float:
        return FieldType(FLOAT, False)
    if t is str:
        return FieldType(STR, False)
    return FieldType(OTHER, True)


def _function_type(name: str, args: list) -> FieldType:
    any_nullable = any(a.nullable for a in args)
    if name in _STR_FUNCS:
        return FieldType(STR, any_nullable)
    if name in ("abs", "round"):
        return FieldType(args[0].base if args else OTHER, any_nullable)
    if name == "concat":
        return FieldType(STR, False)
    if name in ("coalesce", "nullif"):
        return FieldType(args[0].base if args else OTHER, True)
    return FieldType(OTHER, True)


def referenced_tables(node: Any) -> list:
    if isinstance(node, TableScan):
        return [node.table]
    if isinstance(node, (SubqueryAlias, Filter, LookupJoin)):
        return referenced_tables(node.input)
    if isinstance(node, (CrossJoin, Join)):
        return referenced_tables(node.left) + referenced_tables(node.right)
    return []
