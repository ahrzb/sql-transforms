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

from sql_transform import _codegen_plan as cp
from sql_transform import _codegen_runtime as rt
from sql_transform._codegen_plan import UnsupportedInCodegen

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
    else:
        raise UnsupportedInCodegen(f"cannot compile {type(node).__name__}")


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
        # Lookup specs go unused until joins land (Task 9).
        plan, _ = cp.optimize(plan, set(static_tables))

        row_schemas = {n: cp.schema_from_pydantic(m) for n, m in row_tables.items()}
        static_schemas = {n: cp.schema_from_arrow(t) for n, t in static_tables.items()}
        validation = cp.validate_columns(
            plan, set(row_tables), row_schemas, static_schemas
        )
        schemas = validation.effective_schemas

        inferred = [(alias, cp.infer_type(e, schemas)) for alias, e in plan.projection]
        for alias, ft in inferred:
            if cp.is_container(ft.base):
                raise UnsupportedInCodegen(
                    f"column '{alias}' is a struct/list, which codegen does not "
                    "support yet"
                )

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
                        row[col] = getattr(row_obj, col)
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
