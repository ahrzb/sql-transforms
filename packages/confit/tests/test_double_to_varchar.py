"""DOUBLE -> VARCHAR text, sign included.

DuckDB renders a double with `duckdb_fmt::format("{}", v)` (v1.5.5,
src/common/operator/string_cast.cpp). The bundled fmt float writer reads
`std::signbit` BEFORE it branches on finiteness -- its own comment says
"value < 0 is false for NaN so use signbit" -- so the sign is prefixed to
`nan` exactly as it is to `inf` and to a zero. A NaN carrying the sign bit
therefore prints `-nan`.

Every string-producing path in this file shares one formatter, so they are
checked together: a fix that reached only the explicit CAST would leave `||`
and `concat` spelling the same value differently.

Casting to text is also the ONLY way the sign is checkable from here: the
comparison contract compares doubles by `repr`, which spells both NaNs
`nan`, so a DOUBLE output column cannot show a flipped sign to any gate.
That is a named blind spot -- the "NaN sign and payload" row of
docs/oracle/10-campaign-validity-and-blind-spots.md (claim: blind-spots,
claim: repr-equality) -- not an oversight in `confit.compare`.

Which NaN a libm hands back is the platform's business, so nothing here pins
a libm-produced sign as a constant -- those cases assert engine == oracle and
no more. The two NaN signs that ARE deterministic get pinned: DuckDB parses
`'nan'` / `'-nan'` through fast_float, which builds the value from
`std::numeric_limits<double>::quiet_NaN()` and a sign flip, and unary minus
is IEEE negation.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, compare

SCHEMA = pa.schema([pa.field("d", pa.float64(), nullable=True)])

NAN = float("nan")
INF = float("inf")

# Every sign/finiteness combination the formatter distinguishes, plus the two
# forms whose exponent spelling the same code already owed DuckDB.
DOUBLES = [NAN, -NAN, INF, -INF, -0.0, 0.0, 1.0, 1e300, 1e-5, None]

# The explicit cast, the two implicit ones (`||` and `concat` bind their
# double operand through the same cast), and a struct field, which is the
# shape the divergence was found in (below).
SQL = (
    "SELECT CAST(d AS VARCHAR) AS c,"
    " d || '|' AS bar,"
    " concat(d, '|') AS cat,"
    " struct_pack(v := CAST(d AS VARCHAR)) AS s"
    " FROM __THIS__"
)

# The same rendering over a NEGATED operand. Unary minus is IEEE negation on
# DuckDB (`-(TR)input` in NegateOperator, v1.5.5), which flips the sign bit of
# a NaN like any other; a lowering that subtracts from a zero instead cannot,
# because IEEE subtraction propagates the operand's NaN sign unchanged.
NEG_SQL = (
    "SELECT CAST(-d AS VARCHAR) AS c,"
    " (-d) || '|' AS bar,"
    " concat(-d, '|') AS cat,"
    " struct_pack(v := CAST(-d AS VARCHAR)) AS s"
    " FROM __THIS__"
)

# The case that found it, verbatim: a NaN out of `pow(-0.25, 0.1)` rendered
# inside a struct field, which the differential fuzz campaign (fuzz/runner.py
# over seeds 0-1999, seed 1804) graded DIVERGE_VALUE -- the engine wrote
# `nan` into the field where DuckDB wrote `-nan`.
FOUND_BY_SQL = (
    "SELECT struct_pack(f0 := CAST(pow(-0.25e0, 0.1e0) AS VARCHAR)) AS s FROM __THIS__"
)

# Both NaN signs without a libm in the way: fast_float spells the parse, and
# IEEE negation spells the flip.
PINNED_SQL = (
    "SELECT CAST(CAST('-nan' AS DOUBLE) AS VARCHAR) AS parsed,"
    " CAST(-CAST('nan' AS DOUBLE) AS VARCHAR) AS negated,"
    " CAST(-CAST('-nan' AS DOUBLE) AS VARCHAR) AS twice"
    " FROM __THIS__"
)


def _force(backend, monkeypatch):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)


def _both(sql, rows, oracle, backend):
    """(engine rows, oracle rows) for `sql` over a one-column DOUBLE table."""
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": SCHEMA}, static_tables={})
    assert fn.backend == backend
    got = fn.infer_rows(rows)
    oracle.table("__THIS__", "d DOUBLE", [(r["d"],) for r in rows])
    return got, compare.rows(oracle.answer(sql))


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.parametrize("sql", [SQL, NEG_SQL], ids=["plain", "negated"])
def test_double_to_varchar_carries_the_sign(sql, backend, monkeypatch, oracle):
    _force(backend, monkeypatch)
    got, want = _both(sql, [{"d": v} for v in DOUBLES], oracle, backend)
    compare.assert_rows(got, want, ordered=True, ctx=sql)


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_folded_negative_nan_in_struct_pack(backend, monkeypatch, oracle):
    _force(backend, monkeypatch)
    got, want = _both(FOUND_BY_SQL, [{"d": 1.0}], oracle, backend)
    compare.assert_rows(got, want, ordered=True, ctx=FOUND_BY_SQL)
    # `pow`'s NaN sign is the platform libm's to choose, so only the shape is
    # stated: whichever sign it picks must have reached the text.
    assert want[0]["s"]["f0"] in {"nan", "-nan"}


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_parsed_and_negated_nan_signs_are_pinned(backend, monkeypatch, oracle):
    _force(backend, monkeypatch)
    got, want = _both(PINNED_SQL, [{"d": 1.0}], oracle, backend)
    compare.assert_rows(got, want, ordered=True, ctx=PINNED_SQL)
    assert want == [{"parsed": "-nan", "negated": "-nan", "twice": "nan"}]
