"""The string-builder budget and pad/repeat counts.

Split out of test_known_divergences.py 2026-08-16; see README.md for what
belongs here (kept behaviour + its ground) versus in
../test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from confit import DuckDBInferFn

# Fuzz campaign 2026-08-11, 169 of 963 findings. DuckDB's lpad and
# rpad take INTEGER, and its binder does NOT implicitly downcast: a BIGINT
# count -- a row column, or even 2::BIGINT -- is a binder error there, while
# this engine's single integer width bound it happily and served what the
# oracle refuses. The count now binds only when it is spelled a way DuckDB
# types INTEGER or narrower: an int32-range literal (possibly under
# +,-,*,%,parens), or an EXPLICIT cast to INTEGER or narrower -- the
# documented spelling for a column count. A bare column or a BIGINT cast
# refuses. repeat and substr take BIGINT on DuckDB and are untouched.

_PAD_SCHEMA = pa.schema(
    [
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=False),
    ]
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT lpad(s, k, 'x') AS o FROM __THIS__",
        "SELECT rpad(s, k, 'x') AS o FROM __THIS__",
        "SELECT lpad(s, CAST(2 AS BIGINT), 'x') AS o FROM __THIS__",
        "SELECT lpad(s, 3000000000, 'x') AS o FROM __THIS__",
    ],
)
def test_a_bigint_pad_count_refuses_like_duckdb(sql, oracle):
    with pytest.raises(ValueError, match="lpad|rpad"):
        DuckDBInferFn(sql, row_tables={"__THIS__": _PAD_SCHEMA}, static_tables={})

    oracle.table("__THIS__", "k BIGINT, s VARCHAR", [(2, "ab")])
    with pytest.raises(Exception, match="No function matches|out of range"):
        oracle.execute(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT lpad(s, 4, 'x') AS o FROM __THIS__",
        "SELECT lpad(s, (1 + 3), 'x') AS o FROM __THIS__",
        "SELECT rpad(s, -2, 'x') AS o FROM __THIS__",
        "SELECT lpad(s, CAST(k AS INTEGER), 'x') AS o FROM __THIS__",
        "SELECT repeat(s, k) AS o FROM __THIS__",
    ],
)
def test_integer_shaped_counts_still_bind_and_match(sql, oracle):
    """The spellings DuckDB types INTEGER keep building — and keep matching:
    literals, constant arithmetic, negatives; repeat's count is BIGINT on
    DuckDB and stays column-friendly."""
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _PAD_SCHEMA}, static_tables={})
    rows = [{"k": 2, "s": "ab"}]
    got = fn.infer_rows(rows)

    oracle.table("__THIS__", "k BIGINT, s VARCHAR", [(2, "ab")])
    want = oracle.answer(sql).to_pylist()
    assert got == want, f"{got} != {want}"


# Follow-up, certification campaign 2026-08-11, seed 1589: the
# count check ran AFTER the NULL short-circuit, so a bare-NULL string let a
# BIGINT count slip through -- lpad(NULL, c1, 'x') served NULL where DuckDB
# still binder-errors on the count. The count check now runs first. A NULL
# string with an INTEGER-shaped count stays served: DuckDB types that
# VARCHAR (measured -- lpad has no BLOB overload, unlike repeat).


def test_a_null_string_does_not_smuggle_a_bigint_pad_count(oracle):
    with pytest.raises(ValueError, match="lpad"):
        DuckDBInferFn(
            "SELECT lpad(NULL, k, 'x') AS o FROM __THIS__",
            row_tables={"__THIS__": _PAD_SCHEMA},
            static_tables={},
        )
    oracle.table("__THIS__", "k BIGINT, s VARCHAR", [(2, "ab")])
    with pytest.raises(Exception, match="No function matches"):
        oracle.execute("SELECT lpad(NULL, k, 'x') AS o FROM __THIS__")


def test_a_null_string_with_an_integer_count_still_serves():
    fn = DuckDBInferFn(
        "SELECT lpad(NULL, 3, 'x') AS o, rpad(NULL, 3, 'x') AS p FROM __THIS__",
        row_tables={"__THIS__": _PAD_SCHEMA},
        static_tables={},
    )
    got = fn.infer_rows([{"k": 2, "s": "ab"}])
    assert got == [{"o": None, "p": None}]


# Fuzz rounds 1+2, ~7 findings + the campaign timeouts. A literal
# pad/repeat count that can exceed the engine's 1 GiB string-builder budget
# refuses at build by name. Data-driven counts (a column, CAST(k AS INTEGER))
# keep the documented runtime cap.
#
# THE GROUND, restated 2026-08-16 because the old one was false. This block
# used to say DuckDB "sometimes serves the 2GB result and sometimes errors",
# making the refusal sound forced by their instability. Measured, DuckDB is
# entirely deterministic:
#
#   repeat('a', n)   n <= 4294967295   serves (2 GiB took 9.0s)
#                    n >  4294967295   Out of Range Error, every time, 0.00s
#   lpad/rpad        n >  2147483647   Binder Error - their count parameter
#                                      is declared INTEGER (the pin above,
#                                      a different fact entirely)
#
# Two deterministic errors for two unrelated reasons; no coin flip, no
# spelling-dependence. So the honest ground is OURS and it is a judgement, not
# a forced hand: a serving engine does not allocate a gigabyte per row. We
# refuse where DuckDB would serve, which the match-or-refuse contract permits
# — and a build-time refusal beats discovering it per-row in production.
#
# Stating it accurately matters beyond tidiness: the same false claim was
# baked into the user-facing refusal text at frontend.rs, telling authors
# DuckDB was unreliable when it is not.

_SB_SCHEMA = pa.schema(
    [
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=False),
    ]
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT lpad(s, 2000000000, 'x') AS o FROM __THIS__",
        "SELECT rpad(s, 1500000000, 'x') AS o FROM __THIS__",
        "SELECT repeat(s, 2000000000) AS o FROM __THIS__",
    ],
)
def test_a_budget_breaking_literal_count_refuses(sql):
    with pytest.raises(ValueError, match="builder|GiB"):
        DuckDBInferFn(sql, row_tables={"__THIS__": _SB_SCHEMA}, static_tables={})


def test_a_large_but_bounded_count_still_serves_and_matches(oracle):
    sql = (
        "SELECT length(lpad(s, 100000, 'x')) AS o,"
        " length(repeat(s, 50000)) AS p FROM __THIS__"
    )
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _SB_SCHEMA}, static_tables={})
    got = fn.infer_rows([{"k": 1, "s": "ab"}])

    oracle.table("__THIS__", "k BIGINT, s VARCHAR", [(1, "ab")])
    want = oracle.answer(sql).to_pylist()
    assert got == want, f"{got} != {want}"
