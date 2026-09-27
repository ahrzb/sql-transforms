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


class _FloatStructUdf:
    """A pure struct UDF over one DOUBLE: None on a NULL argument."""

    name = "fu"
    takes = pa.schema([("x", pa.float64())])
    returns = pa.struct([("f0", pa.int64()), ("f2", pa.float64())])

    def __call__(self, x):
        return None if x is None else (1, x + 0.5)


@pytest.mark.parametrize(
    ("arg", "want_ty"),
    [
        # fuzz seed 40473: the integer-constant cast must finish in the fold,
        # or the call stays at run time and `.f2` keeps its declared DOUBLE.
        ("CAST(NULL AS DOUBLE)", pa.int32()),
        ("CAST(34 AS DOUBLE)", pa.float64()),
    ],
)
def test_a_udf_over_a_cast_integer_constant_folds_at_bind(arg, want_ty, oracle):
    sql = f"SELECT (fu({arg})).f2 AS o0 FROM __THIS__"
    u = _FloatStructUdf()
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _D_SCHEMA}, static_tables={}, udfs=[u]
    )
    got = fn.infer_rows([{"x": 1.0}])

    oracle.create_function(
        "fu",
        lambda x: None if (r := u(x)) is None else {"f0": r[0], "f2": r[1]},
        ["DOUBLE"],
        duckdb.struct_type({"f0": "BIGINT", "f2": "DOUBLE"}),
        null_handling="special",
    )
    oracle.table("__THIS__", "x DOUBLE", [(1.0,)])
    want = oracle.answer(sql)
    assert want.schema.field("o0").type == want_ty, "oracle moved -- remeasure"
    compare.assert_schema(fn.output_schema, want.schema, ctx=sql)
    compare.assert_rows(got, compare.rows(want), ctx=sql)


@pytest.mark.parametrize(
    "expr",
    [
        # fuzz seed 16617, shrunk: floor() over a constant NULL folds to NULL
        # at bind, so the division is NULL and the trapping CASE never runs.
        "(CASE WHEN 0.75e0 THEN CAST('' AS DOUBLE) END) "
        "/ floor(TRY_CAST('%_' AS DOUBLE))",
        "CAST('' AS DOUBLE) / sqrt(TRY_CAST('x' AS DOUBLE))",
        "floor(TRY_CAST('x' AS DOUBLE)) || 'a'",
        "floor(TRY_CAST('x' AS DOUBLE)) + 1",
        "ceil(CAST(NULL AS DOUBLE))",
    ],
)
def test_a_math_call_over_a_constant_null_folds_to_null(expr, oracle):
    sql = f"SELECT {expr} AS o0 FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _D_SCHEMA}, static_tables={})
    got = fn.infer_rows([{"x": 1.0}, {"x": None}])

    oracle.table("__THIS__", "x DOUBLE", [(1.0,), (None,)])
    want = oracle.answer(sql)
    compare.assert_schema(fn.output_schema, want.schema, ctx=sql)
    compare.assert_rows(got, compare.rows(want), ctx=sql)
