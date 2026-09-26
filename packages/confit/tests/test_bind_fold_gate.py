"""The bind-time fold folds only what DuckDB's binder folds."""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn, compare

# Bind-time folding folds only what DuckDB's binder folds.
#
# Our `fold` dead-arm-eliminates a CASE whose column sits in an untaken arm;
# DuckDB's binder refuses to fold anything holding a column at all. An
# arithmetic operand spelled `CASE WHEN false THEN <double column> END` must
# therefore reach the `||` binder live, so `||` stays VARCHAR (no SQLNULL
# collapse) and `abs(...)` / unary minus over it refuse, as DuckDB's binder
# does. Values agree either way (every row is NULL): only the schema and the
# two consumers can see a difference. Parametrized over the operators that
# share the fold (`fold_operand`).
#
# The literal spelling -- `CASE WHEN false THEN 1.0e0 END` -- is NOT here.
# DuckDB's binder folds that one too, so both engines collapse and agree;
# it is pinned as settled behaviour in known_divergences/test_literal_typing.
_D_SCHEMA = pa.schema([pa.field("x", pa.float64())])
_DEAD_ARM = "(CASE WHEN false THEN x END)"


@pytest.mark.parametrize(
    "expr",
    [
        f"(- {_DEAD_ARM}) || 'y'",
        f"({_DEAD_ARM} + 0.0e0) || 'y'",
        f"({_DEAD_ARM} - 0.0e0) || 'y'",
        f"({_DEAD_ARM} * 1.0e0) || 'y'",
    ],
)
def test_dead_arm_column_over_folds_past_the_concat_gate(expr, oracle):
    sql = f"SELECT {expr} AS o0 FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _D_SCHEMA}, static_tables={})
    got = fn.infer_rows([{"x": 1.0}])

    oracle.table("__THIS__", "x DOUBLE", [(1.0,)])
    want = oracle.answer(sql)
    compare.assert_schema(fn.output_schema, want.schema, ctx=sql)
    compare.assert_rows(got, compare.rows(want), ctx=sql)


@pytest.mark.parametrize("consumer", ["abs", "-"])
def test_dead_arm_over_fold_serves_calls_duckdb_refuses(consumer, oracle):
    inner = f"(- {_DEAD_ARM}) || 'y'"
    expr = f"- ({inner})" if consumer == "-" else f"{consumer}({inner})"
    sql = f"SELECT {expr} AS o0 FROM __THIS__"

    oracle.table("__THIS__", "x DOUBLE", [(1.0,)])
    with pytest.raises(duckdb.BinderException, match="No function matches"):
        oracle.answer(sql)
    with pytest.raises(ValueError):
        DuckDBInferFn(sql, row_tables={"__THIS__": _D_SCHEMA}, static_tables={})
