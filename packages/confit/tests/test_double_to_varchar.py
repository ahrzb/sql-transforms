"""DOUBLE -> VARCHAR text, sign included.

DuckDB renders a double with `duckdb_fmt::format("{}", v)` (v1.5.5,
src/common/operator/string_cast.cpp). The bundled fmt float writer reads
`std::signbit` BEFORE it branches on finiteness -- its own comment says
"value < 0 is false for NaN so use signbit" -- so the sign is prefixed to
`nan` exactly as it is to `inf` and to a zero. A NaN carrying the sign bit
therefore prints `-nan`, and `pow(-0.25, 0.1)` is one of the many ways a
query reaches one.

Every string-producing path in this file shares one formatter, so they are
checked together: a fix that reached only the explicit CAST would leave `||`
and `concat` spelling the same value differently.
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
# double operand through the same cast), a struct field -- the seed-1804
# shape -- and a length, which is what makes a dropped sign fail even where
# repr-comparison of the strings would not.
SQL = (
    "SELECT CAST(d AS VARCHAR) AS c,"
    " d || '|' AS bar,"
    " concat(d, '|') AS cat,"
    " struct_pack(v := CAST(d AS VARCHAR)) AS s,"
    " length(CAST(d AS VARCHAR)) AS n"
    " FROM __THIS__"
)

# The campaign case verbatim: a constant NaN, folded at build time and only
# then rendered, inside the struct_pack that first showed the divergence.
SEED_1804_SQL = (
    "SELECT struct_pack(f0 := CAST(pow(-0.25e0, 0.1e0) AS VARCHAR)) AS s FROM __THIS__"
)


def _both(sql, rows, oracle, backend):
    """(engine rows, oracle rows) for `sql` over a one-column DOUBLE table."""
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": SCHEMA}, static_tables={})
    assert fn.backend == backend
    got = fn.infer_rows(rows)
    oracle.table("__THIS__", "d DOUBLE", [(r["d"],) for r in rows])
    return got, compare.rows(oracle.answer(sql))


@pytest.fixture
def backend(request, monkeypatch):
    if request.param == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    return request.param


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"], indirect=True)
def test_double_to_varchar_carries_the_sign(backend, oracle):
    rows = [{"d": v} for v in DOUBLES]
    got, want = _both(SQL, rows, oracle, backend)
    compare.assert_rows(got, want, ordered=True, ctx=SQL)


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"], indirect=True)
def test_folded_negative_nan_in_struct_pack(backend, oracle):
    got, want = _both(SEED_1804_SQL, [{"d": 1.0}], oracle, backend)
    compare.assert_rows(got, want, ordered=True, ctx=SEED_1804_SQL)
    # State the value rather than only comparing it: if DuckDB's `pow` ever
    # stops handing back a signed NaN here, this test would otherwise keep
    # passing while checking nothing.
    assert want == [{"s": {"f0": "-nan"}}]
