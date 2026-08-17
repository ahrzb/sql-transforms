"""CAST: rounding mode, and the refusal text about it (TASK-70, TASK-113).

Split out of test_known_divergences.py 2026-08-16; see README.md for what
belongs here (kept behaviour + its ground) versus in
../test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from _helpers import duck
from confit import DuckDBInferFn

# ------------------------------------------------- CAST rounding mode --
#
# `lower::cast` emitted `Inst::Ftoi { mode: RoundMode::Round }` under the
# comment "ftoi.round matches DuckDB CAST rounding". It did not. Both backends
# implemented RoundMode::Round as Rust `f64::round()` — half AWAY from zero —
# while DuckDB's DOUBLE->BIGINT cast is half-to-EVEN.
#
# FIXED 2026-08-08 (TASK-70). The mode is now `RoundMode::Nearest`
# (`ftoi.nearest` in the IR text), half-to-even on both backends. Only CAST
# and TRY_CAST ever emitted it, so no other op moved.
#
# TWO SEPARATE ROUNDINGS LIVE HERE AND THEY ARE EASY TO CONFUSE:
#
#   CAST(DOUBLE AS BIGINT)   half to even        -2.5 -> -2
#   CAST(DECIMAL AS BIGINT)  half away from zero -2.5 -> -3
#   round(DOUBLE)            half away from zero -2.5 -> -3.0
#
# Two pre-existing Rust pins asserted half-away-from-zero for the DOUBLE cast
# and had to be corrected. Both were written from a DuckDB query on a bare
# `-2.5` literal — which DuckDB types DECIMAL(2,1), not DOUBLE. Measure a
# DOUBLE cast with a DOUBLE column or an explicit `::DOUBLE`, never a literal.
# (Decimal literals binding as f64 is a separate, deliberate v0 divergence;
# see docs/known-limitations.md.)

CAST_SCHEMA = pa.schema([pa.field("f", pa.float64(), nullable=False)])
_CAST_F = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 2.6, -2.6, 1e19]


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_cast_double_to_bigint_rounds_half_to_even(backend, monkeypatch):
    """Every exactly-representable half-integer used to differ by 1. `1e19` is
    on the end to keep the range-guarded TRY_CAST path (a second `Ftoi` site)
    in the same comparison — it overflows BIGINT and must become NULL."""
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    sql = "SELECT TRY_CAST(f AS BIGINT) AS i, round(f) AS r FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": CAST_SCHEMA}, static_tables={})
    assert fn.backend == backend
    got = [(r["i"], r["r"]) for r in fn.infer_rows([{"f": v} for v in _CAST_F])]
    want = duck(sql, "CREATE TABLE __THIS__ (f DOUBLE)", [(v,) for v in _CAST_F])
    assert got == want
    # And the contrast, stated rather than implied: the two columns disagree
    # on every tie, so this test fails if the cast ever adopts round()'s mode.
    assert [(i, r) for i, r in got if i is not None and float(i) != r] != []


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_plain_cast_double_to_bigint_traps_out_of_range(backend, monkeypatch):
    """The non-TRY path shares the rounding but keeps its own range trap."""
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    sql = "SELECT CAST(f AS BIGINT) AS i FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": CAST_SCHEMA}, static_tables={})
    assert fn.backend == backend
    fine = [v for v in _CAST_F if v != 1e19]
    got = [r["i"] for r in fn.infer_rows([{"f": v} for v in fine])]
    want = [
        r[0]
        for r in duck(sql, "CREATE TABLE __THIS__ (f DOUBLE)", [(v,) for v in fine])
    ]
    assert got == want
    with pytest.raises(ValueError, match="range"):
        fn.infer_rows([{"f": 1e19}])


# TASK-113 AC #1: the refusal text must not assert DuckDB errors on an input
# where DuckDB serves a value. Measured 2026-08-15: a numeric string that
# fits the target is parsed and ROUNDED by DuckDB (both CAST and TRY_CAST);
# only a non-numeric string, or one whose value misses the target's range,
# actually errors there. The cast itself is still refused — this is about
# the message telling the truth, not about implementing the cast.
# ---------------------------------------------------------------------------
_CAST_ROW = pa.schema([pa.field("k", pa.int64(), nullable=False)])


def _cast_refusal(expr: str) -> str:
    with pytest.raises(ValueError) as e:
        DuckDBInferFn(
            f"SELECT {expr} AS o FROM __THIS__",
            row_tables={"__THIS__": _CAST_ROW},
            static_tables={},
        )
    return str(e.value)


@pytest.mark.parametrize(
    "expr",
    [
        "CAST('1.5' AS BIGINT)",
        "CAST('2.5' AS BIGINT)",
        "CAST('-1.5' AS BIGINT)",
        "CAST('1e2' AS BIGINT)",
        "CAST('.5' AS BIGINT)",
        "CAST('1.5' AS TINYINT)",
    ],
)
def test_refusal_does_not_claim_duckdb_errors_when_it_serves(expr):
    """DuckDB answers a value for every one of these."""
    msg = _cast_refusal(expr)
    assert "errors at plan time" not in msg, msg
    # TRY_CAST is not the escape hatch here either — it returns the same
    # rounded value, so pointing at it would be a second false claim.
    assert "TRY_CAST is the NULL-yielding spelling" not in msg, msg
    assert "round" in msg.lower(), msg


@pytest.mark.parametrize(
    "expr",
    [
        "CAST('abc' AS BIGINT)",
        "CAST('' AS BIGINT)",
        "CAST('99999999999999999999' AS BIGINT)",
        "CAST('300' AS TINYINT)",
    ],
)
def test_refusal_keeps_the_true_claim_where_duckdb_really_errors(expr):
    """Non-numeric, or numeric but outside the target — DuckDB's CAST errors
    and TRY_CAST yields NULL, so the original wording is accurate."""
    msg = _cast_refusal(expr)
    assert "TRY_CAST" in msg, msg
