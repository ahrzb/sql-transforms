"""Literal and NULL typing: bare NULLs, INT32 overflow, signed zero.

Split out of test_known_divergences.py 2026-08-16; see README.md for what
belongs here (kept behaviour + its ground) versus in
../test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, compare

# Fuzz campaign 2026-08-11, 11 schema findings + 5 downstream binder splits.
# DuckDB types a bare NULL argument FIRST (INTEGER, or the BLOB overload), and
# lets IT drive the signature -- so nullif(NULL, 84.7e0)
# comes back int32 there and double here, and repeat(NULL, n) is BLOB there
# and string here, which then splits every OUTER call binding a BLOB
# (strpos/ltrim/lower/levenshtein/LIKE -- the campaign's five singleton
# "No function matches" findings). Values all NULL, schemas apart, so
# concat_tables against the oracle raises: the same schema-divergence
# consequence the string-type and integer-width classes have, through a
# different door. The two divergent adopters now refuse by name;
# CAST(NULL AS ...) stays the documented spelling, and adopters that agree
# with DuckDB (upper(NULL), coalesce(NULL, x), nullif(x, NULL)) are
# untouched.

_BN_SCHEMA = pa.schema(
    [
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=False),
    ]
)


@pytest.mark.parametrize(
    "sql",
    [
        # The nullif face closed with m-8 phase 2 (int32 is real; parity
        # pinned in test_integer_widths.py). These two are the BLOB face.
        "SELECT repeat(NULL, 3) AS o FROM __THIS__",
        "SELECT ltrim(repeat(NULL, k)) AS o FROM __THIS__",
    ],
)
def test_a_divergently_typed_bare_null_argument_refuses(sql):
    with pytest.raises(ValueError, match="NULL"):
        DuckDBInferFn(sql, row_tables={"__THIS__": _BN_SCHEMA}, static_tables={})


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT nullif(CAST(NULL AS DOUBLE), 84.754e0) AS o FROM __THIS__",
        "SELECT nullif(84.754e0, NULL) AS o FROM __THIS__",
        "SELECT upper(NULL) AS o FROM __THIS__",
        "SELECT repeat('ab', NULL) AS o FROM __THIS__",
        "SELECT coalesce(NULL, 2.5e0) AS o FROM __THIS__",
    ],
)
def test_agreeing_null_adopters_still_bind_and_match(sql, oracle):
    """Everywhere the adopted type EQUALS DuckDB's inference, bare NULL keeps
    working -- schema compared too, that being the whole point."""
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _BN_SCHEMA}, static_tables={})
    got = fn.infer_arrow(
        pa.table({"k": pa.array([2], pa.int64()), "s": pa.array(["ab"], pa.string())})
    )

    oracle.table("__THIS__", "k BIGINT, s VARCHAR", [(2, "ab")])
    want = oracle.answer(sql)
    compare.assert_schema(got.schema, want.schema, ctx=sql)
    compare.assert_rows(compare.rows(got), compare.rows(want), ordered=True, ctx=sql)


# Fuzz campaign 2026-08-11, 16 DIVERGE_TRAP findings. DuckDB types
# integer literals INTEGER and computes their arithmetic in 32 bits, so
# `-6 * (- 2147483647)` ERRORS there -- while the engine's single i64 width
# served 12884901882 where the oracle traps. Literal-shaped integer
# arithmetic is now evaluated at build in checked int32, DuckDB's own
# semantics, and a subtree that would trap refuses by name. The residual --
# `CAST(k AS INTEGER) * 2` trapping data-dependently at row time -- needed
# the declared-width design and landed with it (the runtime narrow trap is
# pinned in test_integer_widths.py); a BIGINT operand anywhere in the
# expression keeps 64-bit math on both engines.

_OV_SCHEMA = pa.schema([pa.field("k", pa.int64(), nullable=False)])


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT (-6 * (- 2147483647)) AS o FROM __THIS__",
        "SELECT ((2000000000 + 2000000000) - 2000000000) AS o FROM __THIS__",
        "SELECT (2147483647 + 1) AS o FROM __THIS__",
    ],
)
def test_int32_literal_overflow_refuses_where_duckdb_traps(sql, oracle):
    with pytest.raises(ValueError, match="INTEGER|int32|32"):
        DuckDBInferFn(sql, row_tables={"__THIS__": _OV_SCHEMA}, static_tables={})

    oracle.table("__THIS__", "k BIGINT", [(1,)])
    with pytest.raises(Exception, match="[Oo]verflow"):
        oracle.execute(sql).fetchall()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT (-6 * CAST(-2147483647 AS BIGINT)) AS o FROM __THIS__",
        "SELECT (k * 2147483647) AS o FROM __THIS__",
        "SELECT (2 + 3) AS o FROM __THIS__",
        "SELECT (2000000000 % 0 + 1) AS o FROM __THIS__",
    ],
)
def test_bigint_and_in_range_literal_arithmetic_still_matches(sql, oracle):
    """A BIGINT operand keeps 64-bit math on BOTH engines; in-range literal
    arithmetic and the measured INTEGER%0->NULL stay served and matching."""
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _OV_SCHEMA}, static_tables={})
    got = fn.infer_rows([{"k": 2}])

    oracle.table("__THIS__", "k BIGINT", [(2,)])
    want = oracle.answer(sql).to_pylist()
    compare.assert_rows(got, want, ctx=sql)


# Fuzz campaign 2026-08-11, 113 of 963 findings. Unary minus was
# lowered as `0 - x` -- the comment on that lowering even said so -- and IEEE
# `0.0 - 0.0` is +0.0, so the sign of negative zero vanished everywhere it
# could arise: the folded literal `-0.0e0`, runtime `(- x)` at x = 0.0, and
# any product with a signed zero operand fed through the fold. Observable at
# any magnitude through division (the sign of infinity) and as text through
# CAST AS VARCHAR. The fix subtracts from -0.0 for FLOAT operands, which is
# exact IEEE negation for every double; the integer path keeps 0 - x and its
# i64::MIN trap, matching DuckDB.

_NEG_SCHEMA = pa.schema([pa.field("x", pa.float64(), nullable=False)])


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.parametrize(
    ("sql", "rows"),
    [
        ("SELECT -0.0e0 AS o0 FROM __THIS__", [{"x": 1.0}]),
        ("SELECT (- x) AS o0 FROM __THIS__", [{"x": 0.0}]),
        ("SELECT (- (x * -1.5e0)) AS o0 FROM __THIS__", [{"x": -0.0}]),
        ("SELECT (1.0e0 / (x * -0.0e0)) AS o0 FROM __THIS__", [{"x": 1.0}]),
        ("SELECT CAST((- x) AS VARCHAR) AS o0 FROM __THIS__", [{"x": 0.0}]),
    ],
)
def test_negative_zero_keeps_its_sign(sql, rows, backend, monkeypatch, oracle):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _NEG_SCHEMA}, static_tables={})
    got = fn.infer_rows(rows)

    oracle.table("__THIS__", "x DOUBLE", [(r["x"],) for r in rows])
    want = oracle.answer(sql).to_pylist()
    compare.assert_rows(got, want, ctx=sql)
