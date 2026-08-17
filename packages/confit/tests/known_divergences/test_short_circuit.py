"""WHERE short-circuit and three-valued logic (TASK-75).

Split out of test_known_divergences.py 2026-08-16; see README.md for what
belongs here (kept behaviour + its ground) versus in
../test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from _helpers import _node, _tree_udf, duck
from confit import DuckDBInferFn

# ------------------------------------------- WHERE does not short-circuit --
#
# `fn kleene` was "branchless Kleene AND/OR from flag algebra" and emitted
# BOTH operands unconditionally. `fn case`, immediately below it, DOES branch
# — which is why the same trapping call inside a never-taken CASE arm was
# correctly skipped. So a guard that excluded every row still evaluated the
# thing it was written to guard, and its trap killed the whole request.
#
# FIXED 2026-08-08 (TASK-75). The branchless form is kept — it is what makes
# three-valued NULL semantics cheap — and is now used only when the RIGHT
# operand cannot trap, which is the overwhelmingly common case (`a > 1 AND
# b < 2` is still entirely branchless). When it can trap, AND/OR lowers to a
# branch that evaluates the right operand only on rows the left one does not
# already decide: a definite FALSE decides an AND, a definite TRUE decides an
# OR, and a NULL decides nothing — so the right operand still runs there, and
# still traps there, exactly as DuckDB does.
#
# "Can this trap" is `plan::may_trap`, the same predicate the JOIN ON residual
# rule uses (TASK-74). One definition, so the two cannot drift apart.
#
# The branch carries a flag param only when the result is NULLABLE, exactly as
# `FB::case` does. That is not bookkeeping: the null-lane discipline says a
# non-nullable SExpr lowers to a bare payload with no flag anywhere, and
# `emit_stores` asserts it. A first cut of this fix always carried one, which
# passed the entire suite in RELEASE — `debug_assert!` compiles out — and
# panicked on `BETWEEN` in debug. Run the suite against a debug build too.


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT k FROM __THIS__ WHERE k = 0 AND 9223372036854775807 + k > 0",
        "SELECT k FROM __THIS__ WHERE k > 0 OR 9223372036854775807 + k > 0",
        # the guard in the middle of a longer conjunction
        "SELECT k FROM __THIS__ WHERE k > 0 AND k = 0 AND 9223372036854775807 + k > 0",
        # a trapping CAST rather than an overflow
        "SELECT k FROM __THIS__ WHERE k = 0 AND CAST(1e19 * k AS BIGINT) > 0",
    ],
)
@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_where_and_or_short_circuits_like_duckdb(sql, backend, monkeypatch):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    schema = pa.schema([pa.field("k", pa.int64(), nullable=False)])
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    assert fn.backend == backend
    got = [tuple(r.values()) for r in fn.infer_rows([{"k": 1}, {"k": 2}])]
    assert got == duck(sql, "CREATE TABLE __THIS__ (k BIGINT)", [(1,), (2,)])


_3VL_ROWS = [(k, x) for k in (None, 0, 1) for x in (-1.5, 1.5)]


@pytest.mark.parametrize(
    "right",
    [
        "CAST(x AS BIGINT) > 0",  # may trap -> the new branching path
        "x > 0",  # cannot trap -> the original branchless path
    ],
)
def test_short_circuit_preserves_three_valued_logic(right):
    """AC #3: the branchless form was chosen because Kleene NULL semantics
    fall out of flag algebra for free, and the branch must not regress them.

    The full truth table — left in {NULL, FALSE, TRUE} against a right that
    is FALSE and TRUE — evaluated both ways and checked against DuckDB. The
    two parametrisations differ ONLY in which lowering path they take.
    """
    schema = pa.schema(
        [pa.field("k", pa.int64()), pa.field("x", pa.float64(), nullable=False)]
    )
    sql = f"SELECT k, (k > 0 AND {right}) AS aa, (k > 0 OR {right}) AS oo FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    got = [
        tuple(r.values())
        for r in fn.infer_rows([{"k": k, "x": x} for k, x in _3VL_ROWS])
    ]
    want = duck(sql, "CREATE TABLE __THIS__ (k BIGINT, x DOUBLE)", _3VL_ROWS)
    assert got == want
    # The table really does exercise all three left-hand values.
    assert {r[1] for r in got} == {None, True, False}


def test_where_guard_skips_an_unknown_model_trap():
    """AC #2's other half: scoring an id with no model raises, and a guard
    that excludes every row must stop it from ever being called. DuckDB
    cannot be the oracle here — it has no native tree scoring — so the
    assertion is the empty result the guard implies."""
    schema = pa.schema(
        [
            pa.field("k", pa.int64(), nullable=False),
            pa.field("mid", pa.int64(), nullable=False),
            pa.field("x", pa.float64(), nullable=False),
        ]
    )
    sql = "SELECT k FROM __THIS__ WHERE k = 0 AND m(mid, x) > 0"
    fn = DuckDBInferFn(
        sql,
        row_tables={"__THIS__": schema},
        static_tables={},
        udfs=[_tree_udf([_node(0, -1, 0.0, -1, -1, value=1.0)])],
    )
    rows = [{"k": 1, "mid": 999, "x": 0.0}, {"k": 2, "mid": 999, "x": 0.0}]
    assert fn.infer_rows(rows) == []
    # ... and the trap is still real for a row the guard lets through.
    with pytest.raises(ValueError, match="model"):
        fn.infer_rows([{"k": 0, "mid": 999, "x": 0.0}])
