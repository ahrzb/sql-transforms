"""`CodegenSubstraitFn` — a codegen serving engine driven by a Substrait plan.

Takes a `Context` (registered row-table schemas + static tables) and a Substrait
plan; translates the plan into the `_codegen` IR (via `consumer.consume`) and
reuses `CodegenFn`'s backend-agnostic pipeline and `infer()` unchanged.

`from_sql` is the conditional the caller uses to pass SQL instead of a plan: it
produces the Substrait plan through DataFusion (registering the context's tables)
and feeds it to the same consumer — so SQL and a hand-supplied plan are one path.
"""

from __future__ import annotations

import datafusion
import pyarrow as pa
from datafusion import substrait as dss
from pydantic import BaseModel

from sql_transform._codegen import plan as cp
from sql_transform._codegen.engine import CodegenFn
from sql_transform._codegen_substrait.consumer import consume
from sql_transform._codegen_substrait.context import Context

_ARROW = {
    cp.INT: pa.int64(),
    cp.FLOAT: pa.float64(),
    cp.STR: pa.string(),
    cp.BOOL: pa.bool_(),
}


class CodegenSubstraitFn(CodegenFn):
    """Codegen engine whose plan comes from Substrait, not SQL. Same `infer()`."""

    def __init__(
        self,
        context: Context,
        substrait_plan: bytes,
        output_model: type[BaseModel] | None = None,
    ) -> None:
        table_names = set(context.row_tables) | set(context.static_tables)
        plan = consume(substrait_plan, table_names)
        self._finalize(plan, context.row_tables, context.static_tables, output_model)

    @classmethod
    def from_sql(
        cls,
        sql: str,
        context: Context,
        output_model: type[BaseModel] | None = None,
    ) -> CodegenSubstraitFn:
        """Produce the Substrait plan from SQL via DataFusion, then consume it."""
        return cls(context, sql_to_substrait(sql, context), output_model)


def sql_to_substrait(sql: str, context: Context) -> bytes:
    """Serialize `sql` to a Substrait plan, registering the context's tables in a
    fresh DataFusion session so column names/types and table names resolve."""
    ctx = datafusion.SessionContext()
    for name, model in context.row_tables.items():
        empty = pa.RecordBatch.from_pylist([], schema=_arrow_schema(model))
        ctx.register_record_batches(name, [[empty]])
    for name, table in context.static_tables.items():
        ctx.register_record_batches(name, [table.to_batches()])
    try:
        return dss.Serde.serialize_bytes(sql, ctx)
    except Exception as e:  # noqa: BLE001 -- DataFusion raises a bare Exception
        # DataFusion's own Substrait producer can't express this query (e.g.
        # UNNEST). That surface is unreachable through Substrait — defer it so the
        # differential harness skips rather than fails, matching codegen's pattern.
        raise cp.UnsupportedInCodegen(
            f"DataFusion cannot serialize this query to Substrait: {str(e)[:120]}"
        ) from e


def _arrow_schema(model: type[BaseModel]) -> pa.Schema:
    fields = []
    for name, ft in cp.schema_from_pydantic(model).items():
        arrow_t = _ARROW.get(ft.base)
        if arrow_t is None:
            raise cp.UnsupportedInCodegen(
                f"cannot register column '{name}': non-scalar type not supported yet"
            )
        fields.append(pa.field(name, arrow_t, nullable=ft.nullable))
    return pa.schema(fields)
