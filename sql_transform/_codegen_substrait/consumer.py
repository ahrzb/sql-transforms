"""Translate a Substrait plan into the existing `_codegen` plan IR.

Scope (the "basic part"): Read, Filter and the top-level Project, over scalar
expressions — field references, literals, arithmetic/comparison/boolean scalar
functions, a few string builtins, and casts. Anything else raises
`UnsupportedInCodegen`, so the differential harness reports it as a deferred
surface rather than a wrong answer.

Field references in Substrait are POSITIONAL (an index into the emitting rel's
output). We thread an ordered list of `(table, column_name)` down the tree so
each index resolves back to the `Column(table, name)` the IR/emitter expect.
"""

from __future__ import annotations

from typing import Any

from substrait.proto import Plan

from sql_transform._codegen import plan as cp

# Substrait standard-function name -> IR binary-op name (matches plan._BINOPS).
_BINOP = {
    "add": "add",
    "subtract": "sub",
    "multiply": "mul",
    "divide": "div",
    "modulus": "mod",
    "equal": "eq",
    "not_equal": "neq",
    "lt": "lt",
    "gt": "gt",
    "lte": "lte",
    "gte": "gte",
    "and": "and",
    "or": "or",
}
# Substrait scalar function -> IR builtin name (matches engine._BUILTINS).
_FUNC = {
    "upper": "upper",
    "lower": "lower",
    "abs": "abs",
    "coalesce": "coalesce",
}
# Substrait cast target type field -> IR base type.
_CAST = {
    "string": cp.STR,
    "i64": cp.INT,
    "i32": cp.INT,
    "fp64": cp.FLOAT,
    "fp32": cp.FLOAT,
    "bool": cp.BOOL,
}


def consume(plan_bytes: bytes, table_names: Any = ()) -> cp.Plan:
    """Substrait plan bytes -> `cp.Plan`. Column names come from the plan's own
    Read schemas. `table_names` are the registered names (from the Context): a
    Substrait Read folds its table name to lowercase (DataFusion catalog
    behaviour), so we resolve it back case-insensitively."""
    sp = Plan()
    sp.ParseFromString(plan_bytes)

    funcs: dict[int, str] = {}
    for ext in sp.extensions:
        if ext.HasField("extension_function"):
            funcs[ext.extension_function.function_anchor] = ext.extension_function.name

    if not sp.relations:
        raise cp.UnsupportedInCodegen("Substrait plan has no relations")
    root = sp.relations[0].root
    aliases = list(root.names)
    tmap = {n.lower(): n for n in table_names}
    return _project(root.input, aliases, funcs, tmap)


def _project(
    rel: Any, aliases: list[str], funcs: dict[int, str], tmap: dict[str, str]
) -> cp.Plan:
    """The top rel must be a Project; peel it into `Plan(projection, input)`."""
    if not rel.HasField("project"):
        raise cp.UnsupportedInCodegen(
            f"expected a top-level Project, got {rel.WhichOneof('rel_type')}"
        )
    proj = rel.project
    input_rel, cols = _rel(proj.input, funcs, tmap)

    # Substrait Project APPENDS its computed expressions after the input columns;
    # `emit.output_mapping` (when present) selects which of that pool is output.
    pool: list[Any] = [cp.Column(t, n) for t, n in cols]
    pool += [_expr(e, cols, funcs) for e in proj.expressions]

    mapping = (
        list(proj.common.emit.output_mapping)
        if proj.common.HasField("emit")
        else list(range(len(pool)))
    )
    if aliases and len(aliases) != len(mapping):
        raise cp.UnsupportedInCodegen("Substrait output name/column count mismatch")

    projection = []
    for i, idx in enumerate(mapping):
        alias = aliases[i] if aliases else _col_name(pool[idx], i)
        projection.append((alias, pool[idx]))
    return cp.Plan(projection, input_rel)


def _col_name(expr: Any, i: int) -> str:
    return expr.name if isinstance(expr, cp.Column) else f"col{i}"


def _rel(
    rel: Any, funcs: dict[int, str], tmap: dict[str, str]
) -> tuple[Any, list[tuple[str, str]]]:
    """Translate a relational node, returning (IR rel, ordered [(table, col)])."""
    kind = rel.WhichOneof("rel_type")
    if kind == "read":
        return _read(rel.read, tmap)
    if kind == "filter":
        inner, cols = _rel(rel.filter.input, funcs, tmap)
        pred = _expr(rel.filter.condition, cols, funcs)
        return cp.Filter(inner, pred), cols
    raise cp.UnsupportedInCodegen(f"Substrait relation '{kind}' is not supported yet")


def _read(read: Any, tmap: dict[str, str]) -> tuple[Any, list[tuple[str, str]]]:
    if not read.HasField("named_table"):
        raise cp.UnsupportedInCodegen("only named-table reads are supported")
    folded = read.named_table.names[-1]
    table = tmap.get(folded.lower(), folded)
    names = list(read.base_schema.names)
    # A projection selects a column subset (by index into base_schema), in order.
    if read.HasField("projection"):
        picked = [it.field for it in read.projection.select.struct_items]
        names = [names[i] for i in picked]
    return cp.TableScan(table), [(table, n) for n in names]


def _expr(e: Any, cols: list[tuple[str, str]], funcs: dict[int, str]) -> Any:
    kind = e.WhichOneof("rex_type")
    if kind == "selection":
        idx = e.selection.direct_reference.struct_field.field
        table, name = cols[idx]
        return cp.Column(table, name)
    if kind == "literal":
        return cp.Literal(_literal(e.literal))
    if kind == "cast":
        target = e.cast.type.WhichOneof("kind")
        if target not in _CAST:
            raise cp.UnsupportedInCodegen(f"cast to Substrait type '{target}'")
        return cp.Cast(_expr(e.cast.input, cols, funcs), _CAST[target])
    if kind == "scalar_function":
        return _scalar(e.scalar_function, cols, funcs)
    raise cp.UnsupportedInCodegen(f"Substrait expression '{kind}' is not supported yet")


def _scalar(sf: Any, cols: list[tuple[str, str]], funcs: dict[int, str]) -> Any:
    name = funcs.get(sf.function_reference)
    if name is None:
        raise cp.UnsupportedInCodegen(
            f"unknown Substrait function anchor {sf.function_reference}"
        )
    args = [_expr(a.value, cols, funcs) for a in sf.arguments if a.HasField("value")]
    if name == "not":
        return cp.Not(args[0])
    if name in _BINOP:
        if len(args) != 2:
            raise cp.UnsupportedInCodegen(f"{name} with {len(args)} args")
        return cp.BinaryOp(_BINOP[name], args[0], args[1])
    if name in _FUNC:
        return cp.Func(_FUNC[name], args)
    raise cp.UnsupportedInCodegen(f"Substrait function '{name}' is not supported yet")


def _literal(lit: Any) -> Any:
    kind = lit.WhichOneof("literal_type")
    if kind == "null":
        return None
    if kind in ("fp64", "fp32"):
        return float(getattr(lit, kind))
    if kind in ("i64", "i32", "i16", "i8"):
        return int(getattr(lit, kind))
    if kind == "boolean":
        return bool(lit.boolean)
    if kind == "string":
        return lit.string
    raise cp.UnsupportedInCodegen(f"Substrait literal '{kind}' is not supported yet")
