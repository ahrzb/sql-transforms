"""2026-08-09 differential fuzz sweep: the durable record of its findings.

Six finder agents ran seeded differential campaigns against the engine's
served surfaces (arith, cast, strings, struct, cond, joins + the arrow
sweep; scripts/fuzz/differential.py), then every finding was verified
twice: once by hand with a fresh construction, once by two independent
refuters that had to build their own reproductions and default to
"refuted". Raw candidates: 3 A-class + 2 E-class + 1 C-class survived
dedup; two were dropped by the oracle-artifact carve-out (below), leaving
the four pins in this file. Every finding reproduces on BOTH backends
(cranelift JIT and SPECIALIZER_FORCE_INTERP=1 interpreter); none is a
backend-selection artifact.

The four pins:

* T1  `CAST(-f AS VARCHAR)` on f = 0.0: DuckDB renders `-0.0`, we render
      `0.0`. Mechanism measured, formatter exonerated: `f * -1.0` renders
      `-0.0` correctly on both engines. The unary minus lowers to
      `0 - f`, and under IEEE round-nearest `0.0 - 0.0` is `+0.0`:
      negation loses the sign of zero instead of flipping its bit. The
      loss is observable past the formatter via any operation that
      distinguishes the sign of zero: `1.0 / (-f)` is `-inf` for DuckDB
      and `inf` for us; `pow(-f, -3.0)` likewise (`-inf` vs `inf`).
      f = -0.0, 1.0, -1.0, ±inf all match bit-for-bit; only +0.0
      diverges.

* T2  `CAST(f / f AS VARCHAR)` on f = 0.0: DuckDB renders `-nan`, we
      render `nan`. The raw DOUBLE is bit-identical on both sides
      (0xFFF8...  — the x86 default QNaN, which carries the sign bit);
      only the DOUBLE-to-VARCHAR formatter diverges, and it diverges
      unconditionally: it never renders the sign of a NaN payload,
      where DuckDB's renders it. Refuter probe: `CAST(f * -1.0
      AS VARCHAR)` on f = nan renders `nan` on both engines (DuckDB
      flips the NaN sign bit and then renders the sign it has) — i.e.
      the engine's value path preserves NaN sign exactly like DuckDB's
      does; the gap is text-only. Value-bit parity (the `%`-by-zero
      libm pin) is untouched; this is the VARCHAR surface.

* T3  `TRY_CAST(s AS BIGINT)` on VARCHAR: DuckDB parses the string to
      DOUBLE and rounds half away from zero — `'12.9' -> 13`,
      `'.5' -> 1`, `'-3.7' -> -4`, `'1e3' -> 1000`, `'0x10' -> 16`
      (hex!). We parse strict integer grammar and serve NULL for every
      one of those. `TRY_CAST(s AS DOUBLE)` and `TRY_CAST(s AS VARCHAR)`
      agree on the same strings, so the divergence is confined to the
      string->BIGINT cast.

* T5  `CAST(s AS BIGINT)` on the same strings: DuckDB SERVES the rounded
      values; we build the query (the value can only fail per-row) and
      then raise a Conversion Error at INFERENCE time — a false trap on
      a query DuckDB serves, the class-C breach. Same root cause as T3
      (strict integer string parser) with the non-TRY failure surface.

Refuted and NOT pinned (behind the docs/known-limitations.md
oracle-artifact carve-outs, not engine bugs):

* AND/OR/BETWEEN eager-evaluation traps. Two candidates —
  `k > 0 AND log2(k) > 0` and `NOT (1 BETWEEN k2 AND ceil(-k))` on
  INT64_MIN — DuckDB raises "cannot take logarithm of a negative
  number" / "Overflow in negation" on MIXED batches and SERVES the very
  same queries on singleton batches; its own constant fold serves
  without error. The trap flips with sibling-row composition, the same
  family as the ILIKE NUL statistics-dependent kernel and the
  "DuckDB SELF-inconsistency (row path vs constant fold)" rows of
  docs/known-limitations.md. A row-at-a-time engine cannot reproduce a
  batch-shaped evaluation order even in principle; our lazy
  evaluation matches DuckDB's singleton and constant-fold semantics.

* substr negative-start underflow. DuckDB's kernel selection
  (char-positioned vs byte-positioned) flips with sibling rows
  (multibyte string anywhere in the batch switches the whole batch to
  the byte kernel, changing the answer for ASCII rows too), and its
  constant fold of nested substr disagree with its own runtime kernel.
  Same two carve-outs as above. On singleton ASCII batches we match
  DuckDB bit-for-bit.

Provenance: 6 finder agents over 7 surfaces (~60k seeded cases with the
existing standing fuzzer's seeds re-run at depth), two independent
refuters per finding, each required to build its own construction and to
default to "refuted". Every pin above survived both refutations; every
find either survived with measured bounds (T1, T3, T5) or was
re-scoped to exactly what the evidence supported (T2: value-path
exonerated, formatter-only).
"""

from __future__ import annotations

import duckdb
import pytest
from confit import DuckDBInferFn
from pydantic import create_model

# --------------------------------------------------------------- helpers --


def duck(sql: str, ddl: str, rows: list[list]) -> list[tuple]:
    con = duckdb.connect()
    con.execute(ddl)
    for r in rows:
        con.execute(f"INSERT INTO __THIS__ VALUES ({', '.join('?' * len(r))})", list(r))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def engine(sql: str, model, rows: list[dict]):
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": model}, static_tables={}, output="dict"
    )
    return [
        tuple(r.values()) for r in fn.infer({"__THIS__": [model(**r) for r in rows]})
    ]


# --------------------------------------------------- T1: negation and -0.0 --
#
# Unary minus on a DOUBLE column lowers to `0 - f`. That is fine for every
# value except +0.0: under IEEE round-nearest, 0.0 - 0.0 = +0.0, while
# DuckDB's negation flips the sign bit and serves -0.0. The sign of zero
# is observable through any operation that respects it, so this is not a
# display artifact: 1.0 / (-0.0) is -inf in DuckDB and +inf for us, and
# pow(-0.0, -3) is -inf vs +inf.


@pytest.mark.xfail(
    strict=True,
    reason="Unary minus on f64 lowers to 0 - f, which under IEEE round-nearest "
    "turns -0.0 into +0.0 (0.0 - 0.0 = +0.0). DuckDB flips the sign bit "
    "instead (f * -1.0 renders -0.0 correctly on both engines, so the "
    "formatter is exonerated). Observable past the formatter: "
    "1.0 / (-f) is -inf in DuckDB and +inf for us; pow(-f, -3.0) is "
    "-inf vs +inf. All other input values (-0.0, 1.0, -1.0, ±inf) match "
    "bit-for-bit; only +0.0 diverges. Found by the 2026-08-09 differential "
    "fuzz sweep (arith surface, seeds 1-5); reproduced by two independent "
    "refuters; both backends.",
)
def test_float_negation_serves_negative_zero():
    F = create_model("Row", f=(float, ...))
    sql = (
        "SELECT CAST(-f AS VARCHAR) AS n, "
        "CAST(1.0 / (-f) AS VARCHAR) AS d FROM __THIS__"
    )
    assert engine(sql, F, [{"f": 0.0}]) == [("-0.0", "-inf")]
    assert duck(sql, "CREATE TABLE __THIS__ (f DOUBLE)", [[0.0]]) == [("-0.0", "-inf")]


# ------------------------------------------- T2: NaN sign in the CAST render --
#
# The DOUBLE values are bit-identical (0xFFF8..., x86's default QNaN with
# the sign bit set) on both engines; the divergence is confined to the
# DOUBLE-to-VARCHAR formatter, which never renders the sign of a NaN
# payload where DuckDB's does. The value-path parity claim (the
# `%`-by-zero libm pin in docs/known-limitations.md) is untouched; the
# VARCHAR surface is not.


@pytest.mark.xfail(
    strict=True,
    reason="CAST(f / f AS VARCHAR) on f = 0.0: DuckDB renders '-nan', we "
    "render 'nan'. The underlying DOUBLE is bit-identical on both sides "
    "(0xFFF8... x86 default QNaN carries the sign bit) — the engine's NaN "
    "value path matches DuckDB's; the DOUBLE-to-VARCHAR formatter drops "
    "the sign of a NaN payload unconditionally, DuckDB renders it. "
    "Refuter control: CAST(f * -1.0 AS VARCHAR) on f = nan renders 'nan' "
    "on BOTH engines (DuckDB flips the NaN sign bit and renders the sign "
    "it has), which bounds the defect to the renderer. Found by the "
    "2026-08-09 differential fuzz sweep; both backends.",
)
def test_cast_nan_varchar_renders_sign():
    F = create_model("Row", f=(float, ...))
    sql = "SELECT CAST(f / f AS VARCHAR) AS r FROM __THIS__"
    assert engine(sql, F, [{"f": 0.0}]) == [("-nan",)]
    assert duck(sql, "CREATE TABLE __THIS__ (f DOUBLE)", [[0.0]]) == [("-nan",)]


# --------------------------- T3: TRY_CAST str->BIGINT parses and rounds --
#
# DuckDB's VARCHAR->BIGINT cast materializes through DOUBLE and rounds
# half away from zero, and its string parser accepts fractions, hex and
# scientific notation. Ours accepts strict integer grammar, so TRY_CAST
# serves NULL exactly where DuckDB serves a rounded value. The doubles
# and VARCHAR casts on the same strings agree, bounding the defect to the
# string->integer parse.


@pytest.mark.xfail(
    strict=True,
    reason="TRY_CAST(s AS BIGINT) on VARCHAR: DuckDB parses the string to "
    "DOUBLE, rounds half away from zero and serves a value — '12.9' -> 13, "
    "'.5' -> 1, '-3.7' -> -4, '1e3' -> 1000, '0x10' -> 16 (hex) — while "
    "the engine's string parser accepts strict integer grammar only and "
    "serves NULL for all of them. Controls that agree on the same "
    "strings: TRY_CAST(s AS DOUBLE), TRY_CAST(s AS VARCHAR), and the "
    "strict forms ('13', ' 12', '12 ', '+13') round-trip correctly. Found "
    "by the 2026-08-09 differential fuzz sweep (cast surface, seeds 1-5, "
    "and the arrow sweep); reproduced by two independent refuters with "
    "fresh constructions; both backends.",
)
def test_try_cast_string_to_bigint_rounds_like_duckdb():
    S = create_model("Row", s=(str, ...))
    strings = ["12.9", "13", "abc", "-3.7", "0.5", "1e3", "0x10", ".5", "12."]
    sql = "SELECT TRY_CAST(s AS BIGINT) AS r FROM __THIS__"
    assert [r["r"] for r in engine(sql, S, [{"s": v} for v in strings])] == [
        13,
        13,
        None,
        -4,
        1,
        1000,
        16,
        1,
        12,
    ]
    assert duck(sql, "CREATE TABLE __THIS__ (s VARCHAR)", [[v] for v in strings]) == [
        (13,),
        (13,),
        (None,),
        (-4,),
        (1,),
        (1000,),
        (16,),
        (1,),
        (12,),
    ]


# ------------------------------------------ T5: CAST's false inference trap --
#
# Same strict string parser as T3, non-TRY surface: the query BUILDS
# (the parse failure is per-row, not knowable at build), DuckDB serves
# the rounded values, and we raise a Conversion Error at inference time
# — the class-C breach, a false trap on a query the oracle serves. The
# bind-or-refuse contract has no way to express this, so the honest pin
# is the served-vs-trapped difference itself.


@pytest.mark.xfail(
    strict=True,
    reason="CAST(s AS BIGINT) on VARCHAR builds, and on rows where the "
    "string is strict-integer grammar serves bit-identical values — but on "
    "'12.9', '.5', '0x10', '1e3' etc. it raises a Conversion Error at "
    "INFERENCE time, where DuckDB serves the parsed-and-rounded value "
    "(13, 1, 16, 1000). Same strict-integer string parser as the TRY_CAST "
    "pin above; here the failure surface is a false trap instead of a "
    "NULL. A per-row parse failure cannot be refused at build, so the "
    "engine must either trap — diverging from the served oracle — or "
    "match DuckDB's parse. Found by the 2026-08-09 differential fuzz "
    "sweep; confirmed by an independent refuter's fresh construction; "
    "both backends.",
)
def test_cast_string_to_bigint_serves_instead_of_inference_trap():
    S = create_model("Row", s=(str, ...))
    strings = ["12.9", "13", ".5", "0x10", "1e3"]
    sql = "SELECT CAST(s AS BIGINT) AS r FROM __THIS__"
    want = duck(sql, "CREATE TABLE __THIS__ (s VARCHAR)", [[v] for v in strings])
    assert want == [(13,), (13,), (1,), (16,), (1000,)]
    got = engine(sql, S, [{"s": v} for v in strings])
    assert got == [(13,), (13,), (1,), (16,), (1000,)]
