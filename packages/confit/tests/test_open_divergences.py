"""Divergences we intend to CLOSE — one xfail-strict pin each, ticket named.

It has emptied and refilled inside a single day before -- five pins closed at
once on 2026-08-17 and the campaign that followed the oracle change refilled
it by evening. That is the intended rhythm, not churn: adding a pin here is
how a new divergence gets recorded, and emptying it again is what closing one
looks like.

The split from `known_divergences/` is by INTENT, not by severity:

    known_divergences/   behaviour we have decided to KEEP. Every entry
                         states the ground for keeping it, and its tests
                         PASS — they are regression pins on a settled
                         answer.

    this file            behaviour we have decided to CHANGE. Every entry is
                         xfail(strict=True) and names the task that closes
                         it. When the fix lands the pin flips loudly, and
                         the entry is deleted rather than edited.

Why the separation is worth a second file: mixing the two makes "is this on
purpose?" unanswerable at a glance, and a reader who assumes the wrong one
either implements something we chose not to have, or leaves a real bug
sitting under a paragraph explaining why it is fine. The census on
2026-08-16 found both mistakes already present.

strict=True is the load-bearing part. A pin that silently starts passing is
worse than no pin: it certifies work nobody did.

It emptied on 2026-08-25, when the last two pins closed. The struct leg
FLIPPED -- NATURAL and USING now key on a shared struct exactly like DuckDB
-- and the TIMESTAMP leg became a named REFUSAL rather than a wrong answer
(the decision of 2026-08-25 splits opaque scalar keys off into TASK-134,
still open). The live-oracle pins for both live in test_join_keys.py.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn, compare

# The bind-time fold is stronger than DuckDB's binder, and the `||` collapse
# reads the difference.
#
# Our `fold` dead-arm-eliminates a CASE whose column sits in an untaken arm;
# DuckDB's binder refuses to fold anything holding a column at all. So an
# arithmetic operand spelled `CASE WHEN false THEN <double column> END`
# reaches the `||` binder as a bare NULL here and as a live DOUBLE there --
# and `||`'s bind-time SQLNULL collapse fires on the first and not the
# second. DuckDB answers VARCHAR; we answer an INTEGER-typed SQLNULL, and we
# go on to SERVE `abs(...)` and unary minus over it where DuckDB's binder
# refuses the call outright. Values agree (every row is NULL), so only the
# schema leg -- and the two consumers that stop binding -- can see it.
#
# `bind_foldable` is the gate that exists for exactly this, and both unary
# minus and every binary operator skip it on the shared fold. That is why
# the pin is parametrized over the operators rather than filed against the
# DOUBLE unary minus that surfaced it: one fold, one gate, one fix.
#
# The literal spelling -- `CASE WHEN false THEN 1.0e0 END` -- is NOT here.
# DuckDB's binder folds that one too, so both engines collapse and agree;
# it is pinned as settled behaviour in known_divergences/test_literal_typing.
_D_SCHEMA = pa.schema([pa.field("x", pa.float64())])
_DEAD_ARM = "(CASE WHEN false THEN x END)"


@pytest.mark.xfail(
    strict=True,
    reason="bind-time fold over-folds a dead CASE arm holding a column, "
    "bypassing the || binder's bind_foldable gate; no ticket yet",
)
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


@pytest.mark.xfail(
    strict=True,
    reason="the over-folded NULL types as INTEGER, so these bind here and "
    "DuckDB's binder refuses them; no ticket yet",
)
@pytest.mark.parametrize("consumer", ["abs", "-"])
def test_dead_arm_over_fold_serves_calls_duckdb_refuses(consumer, oracle):
    inner = f"(- {_DEAD_ARM}) || 'y'"
    expr = f"- ({inner})" if consumer == "-" else f"{consumer}({inner})"
    sql = f"SELECT {expr} AS o0 FROM __THIS__"

    oracle.table("__THIS__", "x DOUBLE", [(1.0,)])
    with pytest.raises(duckdb.BinderException, match="No function matches"):
        oracle.answer(sql)
    # We serve it instead, off the INTEGER the over-fold produced.
    with pytest.raises(ValueError):
        DuckDBInferFn(sql, row_tables={"__THIS__": _D_SCHEMA}, static_tables={})
