"""Codegen serving engine — a codegen-based InferFn.

Same contract as the Rust InferFn (sql_transform._interpreter): built from the
post-fit rewritten __STATE__/__THIS__ SQL plus the row/static schemas, .infer()
returns validated Pydantic output rows. The difference is execution -- the plan
is compiled once into a cached Python function, so the per-row path is
straight-line Python over native values instead of an interpreter behind pyo3.

Column references resolve to a direct subscript at compile time rather than a
dict scan per row; that, and not the arithmetic, is where the win comes from.

ponytail: every operation emits a runtime call, so a statically-known int + int
still pays one. Specializing emission off infer_type is the obvious next win --
correctness first, with the differential harness as the net.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, create_model

from sql_transform._codegen import plan as cp
from sql_transform._codegen import runtime as rt
from sql_transform._codegen.plan import UnsupportedInCodegen

__all__ = ["CodegenFn", "UnsupportedInCodegen"]

_OPS = {
    "add": "rt.add",
    "sub": "rt.sub",
    "mul": "rt.mul",
    "div": "rt.div",
    "mod": "rt.mod",
    "eq": "rt.eq",
    "neq": "rt.neq",
    "lt": "rt.lt",
    "gt": "rt.gt",
    "lte": "rt.lte",
    "gte": "rt.gte",
    "and": "rt.and_",
    "or": "rt.or_",
    "dpipe": "rt.dpipe",
}

_BUILTINS = {
    "upper": "rt.upper",
    "lower": "rt.lower",
    "trim": "rt.trim",
    "substr": "rt.substr",
    "substring": "rt.substr",
    "concat": "rt.concat",
    "abs": "rt.abs_",
    "round": "rt.round_",
    "coalesce": "rt.coalesce",
    "nullif": "rt.nullif",
}

_CASTS = {
    cp.STR: "rt.cast_str",
    cp.INT: "rt.cast_int",
    cp.FLOAT: "rt.cast_float",
    cp.BOOL: "rt.cast_bool",
}


class _Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._n = 0

    def var(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def line(self, indent: int, text: str) -> None:
        self.lines.append("    " * indent + text)


def _emit_expr(e: Any, env: dict) -> str:
    if isinstance(e, cp.Column):
        var = env.get(e.table)
        if var is None:
            raise ValueError(f"Unknown table: {e.table}")
        return f"{var}[{e.name!r}]"
    if isinstance(e, cp.Literal):
        return repr(e.value)
    if isinstance(e, cp.BinaryOp):
        return f"{_OPS[e.op]}({_emit_expr(e.left, env)}, {_emit_expr(e.right, env)})"
    if isinstance(e, cp.Not):
        return f"rt.not_({_emit_expr(e.inner, env)})"
    if isinstance(e, cp.Cast):
        return f"{_CASTS[e.target]}({_emit_expr(e.expr, env)})"
    if isinstance(e, cp.Func):
        fn = _BUILTINS.get(e.name)
        if fn is None:
            raise ValueError(f"Unknown function: {e.name}")
        return f"{fn}({', '.join(_emit_expr(a, env) for a in e.args)})"
    if isinstance(e, cp.Case):
        out = _emit_expr(e.default, env)
        for cond, result in reversed(e.arms):
            out = (
                f"({_emit_expr(result, env)} if rt.truthy({_emit_expr(cond, env)}) "
                f"else {out})"
            )
        return out
    raise UnsupportedInCodegen(f"cannot compile {type(e).__name__}")


def _emit_rel(node: Any, env: dict, ind: int, em: _Emitter, body) -> None:
    """Emit `node` as loop/guard levels, calling `body(env, indent)` to fill the
    innermost level. `env` maps each in-scope effective table name to the local
    holding its column dict."""
    if isinstance(node, cp.TableScan):
        v = em.var("_s")
        em.line(ind, f"for {v} in _tables[{node.table!r}]:")
        body({**env, node.table: v}, ind + 1)
    elif isinstance(node, cp.SubqueryAlias):
        real = cp.scan_name(node.input)

        def aliased(inner: dict, i: int) -> None:
            renamed = {k: v for k, v in inner.items() if k != real}
            renamed[node.alias] = inner[real]
            body(renamed, i)

        _emit_rel(node.input, env, ind, em, aliased)
    elif isinstance(node, cp.Filter):

        def filtered(inner: dict, i: int) -> None:
            em.line(i, f"if rt.truthy({_emit_expr(node.predicate, inner)}):")
            body(inner, i + 1)

        _emit_rel(node.input, env, ind, em, filtered)
    elif isinstance(node, cp.CrossJoin):

        def crossed(inner: dict, i: int) -> None:
            _emit_rel(node.right, inner, i, em, body)

        _emit_rel(node.left, env, ind, em, crossed)
    elif isinstance(node, cp.Join):

        def left_done(env_l: dict, i: int) -> None:
            def right_done(env_r: dict, j: int) -> None:
                conds = " and ".join(
                    f"rt.join_eq({_emit_expr(le, env_r)}, {_emit_expr(re, env_r)})"
                    for le, re in node.on
                )
                em.line(j, f"if {conds}:")
                body(env_r, j + 1)

            _emit_rel(node.right, env_l, i, em, right_done)

        _emit_rel(node.left, env, ind, em, left_done)
    elif isinstance(node, cp.LookupJoin):

        def looked_up(inner: dict, i: int) -> None:
            keys = ", ".join(f"rt.key({_emit_expr(k, inner)})" for k in node.keys)
            k = em.var("_k")
            h = em.var("_h")
            em.line(i, f"{k} = ({keys},)")
            em.line(i, f"{h} = _lookups[{node.table!r}].get({k})")
            em.line(i, f"if {h} is None:")
            if node.outer:
                em.line(i + 1, f"{h} = _nullrows[{node.table!r}]")
            else:
                em.line(i + 1, f"raise KeyError(rt.miss({node.table!r}, {k}))")
            body({**inner, node.table: h}, i)

        _emit_rel(node.input, env, ind, em, looked_up)
    else:
        raise UnsupportedInCodegen(f"cannot compile {type(node).__name__}")


def _to_native(v: Any) -> Any:
    """Recursively unwrap a row's Pydantic struct/list value into plain
    dicts/lists. Struct columns can't reach a projected output (rejected as
    UnsupportedInCodegen below), but a JOIN ON key CAN reference one -- and
    two structurally-identical struct columns from different row tables are
    two different synthesized Pydantic classes, so BaseModel.__eq__ (which
    checks type identity) would wrongly call them unequal. Plain dicts
    compare structurally, matching DataFusion/rt.val_eq's semantics."""
    if isinstance(v, BaseModel):
        return v.model_dump()
    if isinstance(v, list):
        return [_to_native(x) for x in v]
    return v


def _build_index(table: Any, key_columns: list) -> tuple:
    """Index a static table by its type-tagged key tuple (mirrors lookup.rs).
    Also returns the all-NULL value row a LEFT lookup miss binds."""
    value_columns = [c for c in table.column_names if c not in key_columns]
    index = {}
    for row in table.to_pylist():
        key = tuple(rt.key(row[c]) for c in key_columns)
        index[key] = {c: row[c] for c in value_columns}
    return index, dict.fromkeys(value_columns)


def compile_plan(plan: cp.Plan) -> tuple:
    em = _Emitter()
    em.line(0, "def _run(_tables, _lookups, _nullrows):")
    em.line(1, "_out = []")

    def project(env: dict, ind: int) -> None:
        items = ", ".join(
            f"{alias!r}: {_emit_expr(e, env)}" for alias, e in plan.projection
        )
        em.line(ind, f"_out.append({{{items}}})")

    _emit_rel(plan.input, {}, 1, em, project)
    em.line(1, "return _out")

    source = "\n".join(em.lines)
    namespace: dict = {"rt": rt}
    exec(compile(source, "<sql_transform.codegen>", "exec"), namespace)  # noqa: S102
    return namespace["_run"], source


class CodegenFn:
    """Codegen counterpart to the Rust InferFn — same constructor and infer()."""

    def __init__(
        self,
        sql: str,
        row_tables: dict,
        static_tables: dict,
        output_model: type[BaseModel] | None = None,
    ) -> None:
        plan = cp.build_plan(sql)
        plan, specs = cp.optimize(plan, set(static_tables))

        row_schemas = {n: cp.schema_from_pydantic(m) for n, m in row_tables.items()}
        static_schemas = {n: cp.schema_from_arrow(t) for n, t in static_tables.items()}
        validation = cp.validate_columns(
            plan, set(row_tables), row_schemas, static_schemas
        )
        schemas = validation.effective_schemas

        inferred = [(alias, cp.infer_type(e, schemas)) for alias, e in plan.projection]

        if output_model is None:
            self.output_model = create_model(
                "OutputRow",
                **{a: (cp.field_type_to_python(ft), ...) for a, ft in inferred},
            )
        else:
            _validate_output_model(output_model, inferred)
            self.output_model = output_model

        self._lookups: dict = {}
        self._nullrows: dict = {}
        for spec in specs:
            table = static_tables.get(spec.static_table)
            if table is None:
                raise ValueError(
                    f"SQL references static table '{spec.static_table}' that was "
                    "not provided"
                )
            self._lookups[spec.static_table], self._nullrows[spec.static_table] = (
                _build_index(table, spec.key_columns)
            )
        self._row_table_columns = validation.row_table_columns
        self._referenced = cp.referenced_tables(plan.input)
        self._run, self.source = compile_plan(plan)

    def infer(self, tables: dict | None = None, **kwargs: list) -> list:
        merged = dict(tables or {})
        merged.update(kwargs)

        value_tables: dict = {}
        for table, rows in merged.items():
            columns = self._row_table_columns.get(table, [])
            out_rows = []
            for row_obj in rows:
                row = {}
                for col in columns:
                    try:
                        row[col] = _to_native(getattr(row_obj, col))
                    except AttributeError as e:
                        raise ValueError(
                            f"Row for table '{table}' is missing attribute '{col}': {e}"
                        ) from e
                out_rows.append(row)
            value_tables[table] = out_rows

        for table in self._referenced:
            if table not in value_tables:
                raise ValueError(f"Unknown table in FROM clause: {table}")

        rows = self._run(value_tables, self._lookups, self._nullrows)
        return [self.output_model.model_validate(r) for r in rows]


def _validate_output_model(model: type[BaseModel], inferred: list) -> None:
    """Reject only what is provably wrong: a missing/extra field vs the
    projection, or a base-type mismatch compatible() cannot excuse. Nullability
    is never a build-time error (mirrors lib.rs validate_output_model)."""
    declared = cp.schema_from_pydantic(model)
    aliases = set()
    for alias, ft in inferred:
        aliases.add(alias)
        if alias not in declared:
            raise ValueError(
                f"output_model is missing field '{alias}' produced by the query"
            )
        if not cp.compatible(ft.base, declared[alias].base):
            raise ValueError(
                f"output_model field '{alias}' is declared as a type incompatible "
                "with the query's inferred output"
            )
    extra = set(declared) - aliases
    if extra:
        raise ValueError(
            f"output_model declares fields not produced by the query: {sorted(extra)}"
        )
