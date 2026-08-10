"""Round-4 fuzz findings — 2026-08-10 differential sweep.

The engine's contract: **either it matches DuckDB bit-for-bit, or it refuses at
build with a named error. There is no third mode.** These are breaches of that
contract found by the standing fuzzer on a fresh run (7 surfaces x 5 seeds x
4000 cases, cranelift + interpreter, `scripts/fuzz/verify_round4.py`):

  E  duck errors, engine serves rows              (should have trapped/refused)
  A  duck ok, engine serves, rows differ          (silent wrong answer)

Each finding reproduced here plays bit-for-bit differently than DuckDB 1.5.5.
As each is fixed its marker comes off and the reason becomes the record of the
fix. `strict=True`: a pin can neither silently start passing nor stop failing.

Confirmed 2026-08-10 from the fuzz candidates (both backends agree on every
case; the divergences live in shared frontend lowering):
"""

from __future__ import annotations

import struct

import duckdb
import pytest
from confit import DuckDBInferFn
from pydantic import create_model

# --------------------------------------------------------------- helpers --


def engine(sql, model, rows, backend, monkeypatch):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": model}, static_tables={}, output="dict"
    )
    assert fn.backend == backend
    return fn.infer({"__THIS__": [model(**r) for r in rows]})


def duck_query(sql, table, tuples):
    con = duckdb.connect()
    con.execute(f"CREATE TABLE __THIS__ ({table})")
    for t in tuples:
        con.execute(f"INSERT INTO __THIS__ VALUES ({','.join('?' * len(t))})", list(t))
    try:
        return ("rows", con.execute(sql).fetchall())
    except BaseException as e:  # noqa: BLE001
        return ("err", str(e).splitlines()[0])
    finally:
        con.close()


def keys(rows):
    """repr-based comparison: `0.0 == -0.0` is True in Python, so value `==`
    cannot see signed-zero differences (the fuzz harness compares `repr`s too)."""
    return [repr(v) for r in rows for v in (r.values() if isinstance(r, dict) else r)]


# =============================================================================
# E class: overflow / domain traps are bypassed inside composed predicates
# =============================================================================
#
# Standalone `-a`, `a + b`, `log2(x)` trap with DuckDB's text (pinned in
# test_duckdb_wave3_mathtail.py). But when the same expression sits INSIDE a
# boolean-residual (an OR/AND composition), the checked instruction is skipped
# and the engine computes the wrapped value silently where DuckDB raises
# `Out of Range Error`. The fuzzer's negation family (38 cases this round).

I64MIN = -9223372036854775808


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.xfail(
    strict=True,
    reason="negation overflow is checked standalone (`-k` traps, mathtail) but "
    "the same `-k` inside `trunc(k) < f OR f BETWEEN -k AND -k` is lowered to a "
    "wrapping subtraction and SERVES where DuckDB raises 'Overflow in negation "
    "of numeric value!'. The checked-arithmetic sites are dropped when the "
    "expression feeds a boolean residual.",
)
def test_composed_negation_overflow_serves(backend, monkeypatch):
    model = create_model("Row", k=(int | None, None), f=(float | None, None))
    rows = [{"k": I64MIN, "f": 1.0}]
    sql = "SELECT (trunc(k) < f OR f BETWEEN -k AND -k) AS r FROM __THIS__"
    assert duck_query(sql, "k BIGINT, f DOUBLE", [(I64MIN, 1.0)])[0] == "err"
    with pytest.raises(Exception, match="Overflow in"):
        engine(sql, model, rows, backend, monkeypatch)


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.xfail(
    strict=True,
    reason="`k + k` on INT64_MAX traps standalone (mathtail pins the text) but "
    "inside `((-f <= f AND b) AND (k + k) BETWEEN ...)` the addition is "
    "computed wrapping and SERVES where DuckDB raises 'Overflow in addition of "
    "INT64'. Same root as the negation pin: checked arithmetic is dropped under "
    "boolean composition.",
)
def test_composed_add_overflow_serves(backend, monkeypatch):
    model = create_model(
        "Row",
        k=(int | None, None),
        k2=(int | None, None),
        f=(float | None, None),
        d=(float | None, None),
        b=(bool | None, None),
    )
    rows = [
        {"k": 64, "k2": -21, "f": -1.5, "d": None, "b": None},
        {
            "k": 9223372036854775807,
            "k2": None,
            "f": float("inf"),
            "d": 1e16,
            "b": False,
        },
        {"k": None, "k2": -1, "f": 284598.43507488724, "d": -1.5, "b": False},
        {
            "k": -1,
            "k2": 9223372036854775807,
            "f": 3.141592653589793,
            "d": None,
            "b": False,
        },
        {"k": 7, "k2": 38, "f": None, "d": float("inf"), "b": None},
        {"k": 2, "k2": 9007199254740992, "f": 3.141592653589793, "d": -1.5, "b": True},
        {"k": None, "k2": 1, "f": 0.1, "d": 2.5e-10, "b": True},
        {"k": -7, "k2": 2, "f": 1e308, "d": None, "b": False},
    ]
    sql = (
        "SELECT ((-f <= f AND b) AND (k + k) BETWEEN (-f * -f) AND (-f * -f)) "
        "AS r, k FROM __THIS__"
    )
    assert (
        duck_query(
            sql,
            "k BIGINT, k2 BIGINT, f DOUBLE, d DOUBLE, b BOOLEAN",
            [
                tuple(r.setdefault(x, None) for x in ("k", "k2", "f", "d", "b"))
                for r in rows
            ],
        )[0]
        == "err"
    )
    with pytest.raises(Exception, match="Overflow in addition"):
        engine(sql, model, rows, backend, monkeypatch)


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.xfail(
    strict=True,
    reason="`log2(-k)` on a negative argument traps standalone but inside "
    "'(NOT (k > 0) AND b) AND log2(-k) BETWEEN 1 AND (k >> -k)' the domain "
    "check is skipped and the engine serves NULL/False rows where DuckDB "
    "raises 'cannot take logarithm of a negative number'. Domain traps are "
    "dropped under the same boolean-residual composition.",
)
def test_composed_log_domain_trap_bypassed(backend, monkeypatch):
    model = create_model(
        "Row",
        k=(int | None, None),
        k2=(int | None, None),
        f=(float | None, None),
        d=(float | None, None),
        b=(bool | None, None),
    )
    rows = [
        {"k": -28, "k2": None, "f": None, "d": 5e-324, "b": False},
        {"k": 1, "k2": 9223372036854775807, "f": -1.5, "d": None, "b": None},
        {"k": 7, "k2": None, "f": 1.2345678901234568e17, "d": 2.5, "b": False},
        {"k": -2147483649, "k2": None, "f": -1.5, "d": -324429.511310842, "b": None},
        {"k": None, "k2": None, "f": -152877.11823959695, "d": -0.0, "b": None},
        {"k": None, "k2": None, "f": 3.141592653589793, "d": None, "b": False},
        {
            "k": None,
            "k2": 67,
            "f": 919535.0783511994,
            "d": 809239.1580474952,
            "b": False,
        },
        {"k": -20, "k2": 0, "f": 0.1, "d": 1.0, "b": True},
    ]
    sql = (
        "SELECT ((NOT (k > 0) AND b) AND log2(-k) BETWEEN 1 AND (k >> -k)) "
        "AS r, k FROM __THIS__"
    )
    assert (
        duck_query(
            sql,
            "k BIGINT, k2 BIGINT, f DOUBLE, d DOUBLE, b BOOLEAN",
            [
                tuple(r.setdefault(x, None) for x in ("k", "k2", "f", "d", "b"))
                for r in rows
            ],
        )[0]
        == "err"
    )
    with pytest.raises(Exception, match="logarithm"):
        engine(sql, model, rows, backend, monkeypatch)


# =============================================================================
# A class: silent value divergences
# =============================================================================


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.xfail(
    strict=True,
    reason="TRY_CAST of a decimal-form string to BIGINT: DuckDB parses '12.9' "
    "(and '12.4' -> 12) and rounds to integers; the engine's string->int cast "
    "requires an integer parse and yields NULL. 145 candidate rows this round "
    "(seed 0..4), all `TRY_CAST(s AS BIGINT)` with decimal strings.",
)
def test_trycast_string_decimal_rounds_in_duck(backend, monkeypatch):
    model = create_model("Row", s=(str | None, None))
    rows = [{"s": "12.9"}, {"s": "12.4"}, {"s": "12.5"}]
    sql = "SELECT TRY_CAST(s AS BIGINT) AS r FROM __THIS__"
    want = duck_query(sql, "s VARCHAR", [(x,) for x in ("12.9", "12.4", "12.5")])
    assert want[0] == "rows"
    assert keys(engine(sql, model, rows, backend, monkeypatch)) == keys(want[1])


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.xfail(
    strict=True,
    reason="Casting a NaN to VARCHAR: DuckDB renders the sign bit ('-nan' for "
    "the IEEE qNaN 0xFFF8000000000000 that BOTH engines compute) while the "
    "engine's float->text drops the NaN sign and renders 'nan'. The value bits "
    "are identical (asserted below); only the served VARCHAR text differs — a "
    "bit-exactness breach on every NaN the engine emits.",
)
def test_float_to_varchar_nan_sign(backend, monkeypatch):
    model = create_model("Row", f=(float, None))
    rows = [{"f": 1.0}]
    sql = "SELECT CAST((f / 0.0) AS VARCHAR) AS r FROM __THIS__"
    got = engine(sql, model, rows, backend, monkeypatch)
    # The IEEE NaN bits agree between the two engines — only the TEXT differs.
    duck_nan = duck_query("SELECT (f / 0.0) AS r FROM __THIS__", "f DOUBLE", [(1.0,)])[
        1
    ][0][0]
    our_nan = confit_nan_bits(backend, monkeypatch)
    assert bits64(duck_nan) == bits64(our_nan)
    assert got == [{"r": "-nan"}]


def confit_nan_bits(backend, monkeypatch):
    model = create_model("Row", f=(float, None))
    out = engine(
        "SELECT (f / 0.0) AS r FROM __THIS__", model, [{"f": 1.0}], backend, monkeypatch
    )
    return out[0]["r"]


def bits64(v: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", v))[0]


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.xfail(
    strict=True,
    reason="Unary float negation loses the zero sign: `-f` is lowered to "
    "`0.0 - f`, and `0.0 - (-0.0)` is `+0.0`, so both `-0.0` and `0.0` come "
    "back as `0.0` where DuckDB's negation flips the sign bit (`-0.0` for "
    "+0.0). Multiplication (`-1.0 * f`) preserves the sign and matches; the "
    "unary-minus lowering is the only site. Fuzzer: ~40 signed-zero rows.",
)
def test_unary_minus_loses_zero_sign(backend, monkeypatch):
    model = create_model("Row", f=(float | None, None))
    rows = [{"f": 0.0}, {"f": -0.0}, {"f": None}]
    sql = "SELECT -f AS r FROM __THIS__"
    want = duck_query(sql, "f DOUBLE", [(0.0,), (-0.0,), (None,)])
    assert want[0] == "rows"
    assert keys(engine(sql, model, rows, backend, monkeypatch)) == keys(want[1])
