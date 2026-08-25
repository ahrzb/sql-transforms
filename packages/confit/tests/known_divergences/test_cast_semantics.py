"""CAST: rounding mode, and the refusal text about it.

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
# FIXED 2026-08-08. The mode is now `RoundMode::Nearest`
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


# The refusal text must not assert DuckDB errors on an input
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
    "expr, want",
    [
        # Since the decimal parser landed (2026-08-19) a numeric-string
        # constant is parsed and rounded HALF AWAY FROM ZERO, exactly as
        # DuckDB's IntegerDecimalCastOperation does. These used to refuse
        # "not implemented".
        ("CAST('1.5' AS BIGINT)", 2),
        ("CAST('2.5' AS BIGINT)", 3),
        ("CAST('-1.5' AS BIGINT)", -2),
        ("CAST('1e2' AS BIGINT)", 100),
        ("CAST('.5' AS BIGINT)", 1),
        ("CAST('1.5' AS TINYINT)", 2),
    ],
)
def test_a_numeric_string_constant_parses_and_rounds(expr, want):
    fn = DuckDBInferFn(
        f"SELECT {expr} AS o FROM __THIS__",
        row_tables={"__THIS__": _CAST_ROW},
        static_tables={},
    )
    assert fn.infer_rows([{"k": 1}]) == [{"o": want}]


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


# ===========================================================================
# DOUBLE -> narrow integers. This engine implements the
# ROUND-FIRST-THEN-CHECK semantics of DuckDB main (PR #24393, merged
# 2026-08-03) and of Postgres. Released DuckDB (v1.5.5, our pinned oracle)
# checks the RAW double first and then hits UB: `CAST(127.5 AS TINYINT)`
# serves a WRAPPED -128 (x86; aarch64 would saturate at INTEGER width), and
# `-128.5` is refused although it rounds into range. An answer that differs
# by CPU is not a function of the query, so we deliberately do NOT reproduce
# it -- the two half-unit slivers below are a KEPT divergence that flips to
# full agreement when the DuckDB pin advances past 1.5.5.
# ===========================================================================
_D2N_ROW = pa.schema([pa.field("x", pa.float64())])


def _d2n(cast: str, val: float, dst: str = "TINYINT"):
    fn = DuckDBInferFn(
        f"SELECT {cast}(x AS {dst}) AS o FROM __THIS__",
        row_tables={"__THIS__": _D2N_ROW},
        static_tables={},
    )
    try:
        return ("S", fn.infer_rows([{"x": val}])[0]["o"])
    except ValueError:
        return ("T", None)


@pytest.mark.parametrize(
    ("val", "cast_want", "try_want"),
    [
        # upstream main's own conformance rows (float_integer_cast.test)
        (126.5, ("S", 126), ("S", 126)),
        (127.4, ("S", 127), ("S", 127)),
        (127.5, ("T", None), ("S", None)),
        (127.6, ("T", None), ("S", None)),
        (-128.4, ("S", -128), ("S", -128)),
        (-128.5, ("S", -128), ("S", -128)),
        (-128.6, ("T", None), ("S", None)),
        (2.5, ("S", 2), ("S", 2)),
        (3.5, ("S", 4), ("S", 4)),
        (float("nan"), ("T", None), ("S", None)),
        (float("inf"), ("T", None), ("S", None)),
        (1e300, ("T", None), ("S", None)),
    ],
)
def test_double_to_narrow_rounds_first_then_checks(val, cast_want, try_want):
    assert _d2n("CAST", val) == cast_want
    assert _d2n("TRY_CAST", val) == try_want


@pytest.mark.parametrize(
    ("val", "duck_155"),
    [
        # the wrap sliver [MAX+0.5, MAX+1): 1.5.5 serves a wrapped value
        (127.5, -128),
        (127.99, -128),
        # the false-refusal sliver (MIN-0.5, MIN): 1.5.5 refuses a fit
        (-128.5, None),
    ],
)
def test_the_155_boundary_slivers_are_a_kept_divergence(val, duck_155):
    """Remeasures the ORACLE each run: while DuckDB 1.5.5 still shows the
    pre-#24393 behaviour these assert the divergence in both directions; the
    day the pin advances, the oracle side flips and this test fails LOUDLY --
    delete it then, the grid above already asserts the agreed semantics."""
    import duckdb as _duck

    con = _duck.connect()
    con.execute("PRAGMA disable_optimizer")
    con.execute("CREATE TABLE t (x DOUBLE)")
    con.execute("INSERT INTO t VALUES (?)", [val])
    try:
        got = con.execute("SELECT CAST(x AS TINYINT) FROM t").fetchall()[0][0]
    except _duck.Error:
        got = None
    assert got == duck_155, f"DuckDB moved past 1.5.5 semantics: {got}"
    # and we deliberately disagree here (fixed semantics, grid above)
    ours = _d2n("CAST", val)
    fixed = ("T", None) if duck_155 == -128 else ("S", -128)
    assert ours == fixed


# ---------------------------------------------------------------------------
# The VARCHAR -> integer grammar, live-oracle. One
# matrix, three widths, both spellings -- if DuckDB's parser moves, this
# remeasures. The grammar is digit-based (see kernels::duck_stoi): rounding
# is half AWAY from zero decided on decimal DIGITS ('1.4999999999999999' is
# 1, not 2), unlike the double path's half-to-even -- DuckDB's two cast
# paths disagree on every exact half and both are preserved here.
# ---------------------------------------------------------------------------
_S2I_ROW = pa.schema([pa.field("x", pa.string())])
_S2I_EDGES = [
    "1",
    "1.0",
    "1.5",
    "2.5",
    "-2.5",
    " 42 ",
    "1e2",
    "1.5e1",
    "0x1A",
    "0X1a",
    "0b101",
    "1_000",
    "1_000.5",
    "  1.5  ",
    "+1.5",
    ".5",
    "1.",
    "0.5",
    "-0.5",
    "+.5",
    "-.5",
    "5.",
    ".5e1",
    "1e+2",
    "1e-0",
    "150e-1",
    "155e-2",
    "145e-2",
    "1.4999999999999999",
    "127.4",
    "127.6",
    "-128.4",
    "-128.5",
    "300",
    "abc",
    "",
    "inf",
    "nan",
    "1e",
    "e5",
    "1e400",
    "0x",
    "0o17",
    "_1",
    "1_",
    "1__0",
    "0x_1A",
    "- 1",
    "2 2",
    "9999999999999999999999",
    "true",
    # 2026-08-24 bounds audit (integer_cast_operator.hpp, v1.5.5 checkout):
    # DuckDB's decimal path accumulates the MANTISSA in int64 (refusing on
    # overflow), silently DROPS fraction digits past int64 capacity, parses
    # the exponent into an int16, and only recognizes 0x/0b on a bare
    # leading zero -- never after a sign. Boundary pairs, matching side
    # first, diverging side second:
    "999999999999999999",  # 18 digits: serves
    "9999999999999999999",  # 19 digits overflow int64: refuses (all widths)
    "9223372036854775807e-18",  # int64::MAX mantissa: serves 9
    "9223372036854775808e-18",  # MAX+1 refuses -- the accumulator IS int64
    "9999999999999999999e-17",
    "12345678901234567890e-16",
    "18446744073709551615e-1",
    "1." + "9" * 37,
    "1." + "9" * 38,  # fraction digits past capacity drop silently: 2
    "1." + "9" * 50,
    "0" * 38 + "1",
    "0" * 39 + "1",  # leading zeros never overflow anything: 1
    "0e39",
    "0e32767",  # int16 exponent, zero mantissa skips the multiply: 0
    "0e32768",  # int16 overflow: refuses
    "1e32767",  # nonzero mantissa overflows in the multiply: refuses
    "1e-10001",
    "1e-32768",  # int16::MIN: serves 0
    "1e-32769",  # refuses
    "-0x10",  # hex/binary need a BARE leading 0 -- a sign refuses
    "+0x10",
    "-0b11",
    "1.5e3",  # exponent past the fraction digits: 1500
    "1.55e1",  # 15.5 rounds away: 16
    "0.5e-1",  # 0.05: 0
    "9.5e-1",  # 0.95: 1
    "1.9e-1",  # 0.19: 0
    "1e1_0",  # underscores live in exponent digits too: 1e10
    "\x0b5",  # vertical tab is space to the trim
    # the exponent is parsed by the same character loop, so its tail grammar
    # is the loop's, not a number's; and the hex/binary loops have no
    # trailing-space skip at all:
    "1e5 ",  # 100000
    "1e5.",  # a bare trailing point in the exponent parses: 100000
    "1e ",  # an all-space exponent is exponent 0: serves 1 (!)
    "5.e2",  # 500
    "1e5.5",  # fraction digits in the exponent refuse
    "0x1A ",  # trailing space after hex refuses...
    " 0x1A",  # ...but leading space is trimmed before dispatch: 26
    "0b1 ",
    "1..5",
    "1.5.5",
]


@pytest.mark.parametrize("dst", ["TINYINT", "INTEGER", "BIGINT"])
@pytest.mark.parametrize("cast", ["CAST", "TRY_CAST"])
def test_string_to_integer_matches_the_oracle(cast, dst):
    import duckdb as _duck

    sql = f"SELECT {cast}(x AS {dst}) AS o FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _S2I_ROW}, static_tables={})
    for v in _S2I_EDGES:
        con = _duck.connect()
        con.execute("PRAGMA disable_optimizer")
        con.execute("CREATE TABLE t (x VARCHAR)")
        con.execute("INSERT INTO t VALUES (?)", [v])
        try:
            want = ("S", con.execute(sql.replace("__THIS__", "t")).fetchall()[0][0])
        except _duck.Error:
            want = ("T", None)
        try:
            got = ("S", fn.infer_rows([{"x": v}])[0]["o"])
        except ValueError:
            got = ("T", None)
        assert got == want, f"{cast}({v!r} AS {dst}): {got} != {want}"
