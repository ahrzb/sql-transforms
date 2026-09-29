"""The bind-time fold folds only what DuckDB's binder folds."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn, compare

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

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


# The same gate holds for an operand that a CAST, a comparison, a UDF call
# or a DECIMAL operator folded on its way up: the column is still in
# DuckDB's tree, so a default-NULL-handling call over it is not replaced by
# a NULL at bind. It runs per row, and a sibling argument that traps still
# traps (none over zero rows).
#
# Measured on DuckDB 1.5.5, optimizer off, PREPARE and EXECUTE apart: the
# binder's fold (function_binder.cpp) walks the arguments in order, skips
# one naming a column, SWALLOWS an argument whose evaluation errors, and
# makes the call NULL on the first one that folds to NULL. So a trapping
# constant loses to a NULL sibling in either order, and there is no
# plan-time error to refuse: a trapping constant fails per row, and not at
# all over zero rows.
_T = {"c1": "str", "c2": "int"}
_T_ROWS = [{"c1": "a", "c2": 1}, {"c1": "b", "c2": 2}]
_BIG = "(9223372036854775807 * 34)"
_DEAD_STR = "CAST((CASE WHEN FALSE THEN c1 END) AS VARCHAR)"
_DEAD_DEC = "(CASE WHEN FALSE THEN CAST(c2 AS DECIMAL(4,2)) END)"


@pytest.mark.parametrize(
    ("expr", "over_rows"),
    [
        (f"repeat(CAST(NULL AS VARCHAR), {_BIG})", "AGREE"),
        (f"repeat(CAST({_BIG} AS VARCHAR), CAST(NULL AS BIGINT))", "AGREE"),
        ("repeat(CAST(NULL AS VARCHAR), CAST('nope' AS BIGINT))", "AGREE"),
        (
            f"repeat(CAST(NULL AS VARCHAR), CASE WHEN TRUE THEN {_BIG} ELSE c2 END)",
            "AGREE",
        ),
        (f"repeat({_DEAD_STR}, {_BIG})", "AGREE_TRAP"),
        (f"repeat({_DEAD_STR}, c2 + 9223372036854775807)", "AGREE_TRAP"),
        (f"repeat({_DEAD_STR}, c2)", "AGREE"),
        # Foldability is read off DuckDB's BOUND tree, not the SQL: a bare
        # NULL makes replace() a constant NULL at bind although it names c1,
        # so the calls around it fold too and the trapping sibling never
        # runs. A "the argument's SQL names no column" test would trap here.
        ("repeat(replace(c1, 'a', NULL), c2 + 9223372036854775807)", "AGREE"),
        ("repeat(upper(replace(c1, 'a', NULL)), c2 + 9223372036854775807)", "AGREE"),
        (
            "lpad(upper(replace(c1, 'a', NULL)), "
            "CAST(c2 AS INTEGER) + 2147483647, 'a')",
            "AGREE",
        ),
        ("repeat('x', CAST('nope' AS BIGINT))", "AGREE_TRAP"),
        ("lpad(CAST(NULL AS VARCHAR), CAST('nope' AS INTEGER), 'a')", "AGREE"),
        ("upper(CAST((9223372036854775807 * 34) AS VARCHAR))", "AGREE_TRAP"),
        (
            "replace(CAST(NULL AS VARCHAR), CAST((9223372036854775807 * 34) "
            "AS VARCHAR), 'a')",
            "AGREE",
        ),
        # The other consumers of the fold: arithmetic's NULL shortcut, ||.
        # (The optimizer folds the dead arm and answers NULL; the oracle is
        # optimizer-off DuckDB.)
        (
            "CAST((CASE WHEN FALSE THEN c1 END) AS BIGINT) "
            "+ (c2 + 9223372036854775807)",
            "DIVERGE_OPT",
        ),
        ("CAST(NULL AS BIGINT) + (c2 + 9223372036854775807)", "AGREE"),
        (f"{_DEAD_STR} || 'y'", "AGREE"),
        (f"abs({_DEAD_DEC}) || 'y'", "AGREE"),
        (f"(-{_DEAD_DEC}) || 'y'", "AGREE"),
        (f"TRY_CAST({_DEAD_DEC} AS DECIMAL(3,1)) || 'y'", "AGREE"),
        (
            f"repeat(CAST(abs({_DEAD_DEC}) AS VARCHAR), c2 + 9223372036854775807)",
            "AGREE_TRAP",
        ),
        (
            f"repeat(CAST((-{_DEAD_DEC}) AS VARCHAR), c2 + 9223372036854775807)",
            "AGREE_TRAP",
        ),
    ],
)
@pytest.mark.parametrize("n_rows", [2, 0])
def test_a_default_null_call_folds_only_over_a_foldable_null(expr, over_rows, n_rows):
    sql = f"SELECT {expr} AS o FROM __THIS__"
    expect = over_rows if n_rows else "AGREE"
    assert_parity(sql, table(_T, _T_ROWS[:n_rows]), expect=expect)


def test_a_decimal_round_precision_must_be_foldable():
    # DuckDB requires round(DECIMAL, n)'s precision to fold at bind; a CASE
    # naming a column does not, even with its arm decided.
    sql = (
        "SELECT round(CAST(c2 AS DECIMAL(9,4)), "
        "CASE WHEN TRUE THEN 1 ELSE {} END) AS o FROM __THIS__"
    )
    rows = table(_T, _T_ROWS)
    v = assert_parity(sql.format("CAST(c2 AS INTEGER)"), rows, expect="REFUSED")
    assert "non-constant precision" in v.detail
    assert_parity(sql.format("2"), rows, expect="AGREE")


_OVERFLOW_CASE = (
    "(CASE WHEN (2.5e0 IS NULL) THEN (-9 + c2) "
    "WHEN ('  pad  ' LIKE '%') THEN (9223372036854775807 * 34) ELSE c2 END)"
)
_SEED_SCHEMA = {"c0": "int32?", "c1": "int32", "c2": "int16?"}
_SEED_ROWS = [{"c0": 20, "c1": 0, "c2": None}, {"c0": 0, "c1": -31, "c2": 1}]


@pytest.mark.parametrize(
    "sql",
    [
        # fuzz seed 18995, as generated.
        "SELECT length('abcdefghijabcdefghijabcdefghij') AS o0, "
        "repeat(CAST((CASE WHEN FALSE THEN c1 END) AS VARCHAR), "
        f"{_OVERFLOW_CASE}) AS o1, 0.0e0 AS o2 FROM __THIS__ "
        "WHERE (('%_' NOT LIKE 'a_c') AND (FALSE OR TRUE))",
        "SELECT repeat(CAST(((CASE WHEN FALSE THEN c1 END) = 1) AS VARCHAR), "
        f"{_OVERFLOW_CASE}) AS o FROM __THIS__",
    ],
)
@pytest.mark.parametrize("n_rows", [2, 0])
def test_a_dead_arm_column_argument_keeps_the_call_live(sql, n_rows):
    rows = table(_SEED_SCHEMA, _SEED_ROWS[:n_rows])
    if n_rows:
        assert_parity(sql, rows, trap="Overflow in multiplication of INT64")
    else:
        assert_parity(sql, rows, expect="AGREE")


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
        # A column-holding argument is not DuckDB-foldable: the call stays
        # at run time even though every row is NULL.
        ("CASE WHEN false THEN x END", pa.float64()),
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
        "sqrt(TRY_CAST('x' AS DOUBLE)) * 2.0e0",
        "TRY_CAST('x' AS DOUBLE) || 'a'",
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
