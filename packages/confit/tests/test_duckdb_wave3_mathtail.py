"""Wave-3 math tail — oracle pins vs DuckDB 1.5.5.

Family: add/subtract/multiply/divide/mod aliases, the // operator, fdiv,
fmod, nextafter. Measured pins live in
packages/confit/docs/specs/2026-07-26-wave3-builtin-pins.md ("Math tail").
Every case below was probed through the vectorized path (table columns).
"""

from __future__ import annotations

import re
import struct

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from test_duckdb_interpreter import duck_check

NAN = float("nan")
INF = float("inf")
I64MAX = 9223372036854775807
I64MIN = -9223372036854775808


# --------------------------------------------------- alias == operator:
# add/subtract/multiply/divide/mod are pure frontend desugars of + - * // %.
# Pairing each alias with its operator in ONE query makes any divergence
# between the two lowerings show up as an intra-row mismatch vs the oracle.


def test_int_aliases_pair_with_operators():
    # (7,2)/(−7,2) witness divide()'s TRUNCATING int division (3 / −3,
    # BIGINT — repr int); mixed-sign rows witness mod's dividend sign
    # (mod(−7,3) = −1, mod(7,−3) = 1). NOTE: DuckDB also has unary
    # add(x)=x / subtract(x)=−x; the engine rejects those cleanly
    # ("unsupported: add with 1 arguments") — out of the pinned surface.
    duck_check(
        "SELECT add(a, b) AS s1, a + b AS s2, subtract(a, b) AS d1, a - b AS d2,"
        " multiply(a, b) AS m1, a * b AS m2, divide(a, b) AS q1, a // b AS q2,"
        " mod(a, b) AS r1, a % b AS r2 FROM __THIS__",
        {"a": "int", "b": "int"},
        [
            {"a": 7, "b": 2},
            {"a": -7, "b": 2},
            {"a": 7, "b": -2},
            {"a": -7, "b": -2},
            {"a": -7, "b": 3},
            {"a": 7, "b": -3},
            {"a": 0, "b": 3},
            {"a": 123456789, "b": 1000},
        ],
    )


def test_double_aliases_pair_with_operators():
    # // on DOUBLE is PLAIN division: −7.5 // 2.0 = −3.75 (NOT floor).
    # mod keeps the dividend's sign on doubles too: mod(−7.5, 2.0) = −1.5.
    duck_check(
        "SELECT add(x, y) AS s1, x + y AS s2, subtract(x, y) AS d1, x - y AS d2,"
        " multiply(x, y) AS m1, x * y AS m2, divide(x, y) AS q1, x // y AS q2,"
        " mod(x, y) AS r1, x % y AS r2 FROM __THIS__",
        {"x": "float", "y": "float"},
        [
            {"x": -7.5, "y": 2.0},
            {"x": 7.5, "y": -2.0},
            {"x": -7.5, "y": 2.5},
            {"x": 7.5, "y": 2.5},
            {"x": 1.5, "y": 3.0},
            {"x": -0.5, "y": 0.25},
        ],
    )


# ------------------------------------------------------- zero divisors:
# ints: // divide % mod all NULL; / promotes to DOUBLE and stays IEEE.
# doubles: // divide NULL, but % mod are NaN VALUES (not NULL) — the
# repr-based compare keeps 'nan' and None apart, so the row is a real pin.


def test_int_zero_divisor_nulls_and_ieee_slash():
    duck_check(
        "SELECT a // b AS q, divide(a, b) AS q2, a % b AS r, mod(a, b) AS r2,"
        " a / b AS f FROM __THIS__",
        {"a": "int", "b": "int"},
        [{"a": 7, "b": 0}, {"a": -7, "b": 0}, {"a": 0, "b": 0}, {"a": 7, "b": 3}],
    )


def test_double_zero_divisor_null_vs_nan_vs_inf():
    # CRITICAL row: x % 0.0 -> NaN value while x // 0.0 -> NULL; / -> ±inf.
    duck_check(
        "SELECT x // y AS q, divide(x, y) AS q2, x % y AS r, mod(x, y) AS r2,"
        " x / y AS f FROM __THIS__",
        {"x": "float", "y": "float"},
        [{"x": 7.5, "y": 0.0}, {"x": -7.5, "y": 0.0}, {"x": 0.0, "y": 0.0}],
    )


# ------------------------------------------------------ overflow traps:
# byte-identical error texts on both lowerings, and WHERE/CASE guards
# genuinely prevent evaluation of the trapping row.


def test_add_overflow_traps_operator_and_alias():
    # Both engines trap on i64::MAX + 1 through both lowerings, with
    # DuckDB's own text verbatim (values interpolated, trailing '!').
    for expr in ["a + b", "add(a, b)"]:
        with pytest.raises(
            Exception,
            match=re.escape("Overflow in addition of INT64 (9223372036854775807 + 1)!"),
        ):
            duck_check(
                f"SELECT {expr} AS s FROM __THIS__",
                {"a": "int", "b": "int"},
                [{"a": I64MAX, "b": 1}],
            )


def test_int64_min_division_overflow_traps():
    # INT64_MIN // −1 and mod(INT64_MIN, −1) both trap in both engines with
    # the DIVISION text for both — DuckDB's own % message says 'division',
    # and unlike add/sub/mul it carries no trailing '!'.
    for expr in ["a // b", "mod(a, b)"]:
        with pytest.raises(
            Exception,
            match=re.escape("Overflow in division of -9223372036854775808 / -1"),
        ):
            duck_check(
                f"SELECT {expr} AS q FROM __THIS__",
                {"a": "int", "b": "int"},
                [{"a": I64MIN, "b": -1}],
            )


def test_overflow_guarded_rows_do_not_trap():
    duck_check(
        "SELECT a + b AS s, add(a, b) AS s2 FROM __THIS__ WHERE a < 100",
        {"a": "int", "b": "int"},
        [{"a": 1, "b": 2}, {"a": I64MAX, "b": 1}],
    )
    duck_check(
        "SELECT CASE WHEN b <> -1 THEN a // b ELSE NULL END AS q FROM __THIS__",
        {"a": "int", "b": "int"},
        [{"a": 7, "b": 2}, {"a": I64MIN, "b": -1}],
    )


# --------------------------------------------------- mod vs fmod signs:
# mod/% = truncated (dividend's sign); fmod = floor pair (DIVISOR's sign).
# Signed-zero split: mod(−7.5, 2.5) = −0.0 but fmod(−7.5, 2.5) = +0.0 —
# repr distinguishes, so one row pins both.


def test_mod_dividend_sign_vs_fmod_divisor_sign():
    duck_check(
        "SELECT mod(x, y) AS m, x % y AS p, fmod(x, y) AS f FROM __THIS__",
        {"x": "float", "y": "float"},
        [
            {"x": -7.5, "y": 2.5},  # m = p = -0.0, f = 0.0
            {"x": -7.5, "y": 2.0},  # m = -1.5, f = 0.5
            {"x": 7.5, "y": -2.0},  # m = 1.5, f = -0.5
        ],
    )


# --------------------------------------------------------- fdiv / fmod:
# the FLOOR-division pair, ALWAYS DOUBLE even on BIGINT inputs (repr float),
# so INT64_MIN fdiv −1 is 9.223372036854776e18 — no trap.


def test_fdiv_fmod_bigint_inputs_yield_double():
    duck_check(
        "SELECT fdiv(a, b) AS q, fmod(a, b) AS r FROM __THIS__",
        {"a": "int", "b": "int"},
        [
            {"a": -7, "b": 2},  # q = -4.0 (floor, not truncate)
            {"a": 7, "b": -3},  # q = -3.0
            {"a": -7, "b": 3},  # r = 2.0 (divisor's sign)
            {"a": 7, "b": 0},  # q = inf, r = nan
            {"a": -7, "b": 0},  # q = -inf
            {"a": I64MIN, "b": -1},  # q = 9.223372036854776e18, NO trap
        ],
    )


def test_fdiv_fmod_double_edges():
    # fmod(1.0, inf) = NaN refutes the C-fmod reading (C gives 1.0): DuckDB
    # computes x − floor(x/y)·y, and floor(0)·inf is NaN.
    duck_check(
        "SELECT fdiv(x, y) AS q, fmod(x, y) AS r FROM __THIS__",
        {"x": "float", "y": "float"},
        [
            {"x": -7.5, "y": 2.0},  # q = -4.0, r = 0.5
            {"x": 7.5, "y": -2.0},  # q = -4.0, r = -0.5
            {"x": -7.5, "y": 2.5},  # q = -3.0, r = +0.0
            {"x": 7.5, "y": 0.0},  # q = inf, r = nan
            {"x": -7.5, "y": 0.0},  # q = -inf, r = nan
            {"x": 1.0, "y": INF},  # q = 0.0, r = nan (C fmod would say 1.0)
            {"x": 0.0, "y": 0.0},  # q = nan, r = nan
        ],
    )


def test_computed_nan_bits_match_oracle():
    # repr collapses every NaN to 'nan', so pin the BITS manually. fmod's
    # NaN comes from hardware arithmetic (0*inf under SSE) and is fff8…
    # on every x86 platform. The %-by-zero NaN comes from LIBM fmod and
    # its SIGN is platform-dependent (Windows ucrt 7ff8…, Linux glibc
    # fff8… — CI-discovered, the cbrt situation again): both engines use
    # the platform libm, so the pin is ENGINE == ORACLE bit agreement,
    # not a constant.
    def bits(v: float) -> str:
        return format(struct.unpack("<Q", struct.pack("<d", v))[0], "016x")

    schema = pa.schema(
        [
            pa.field("x", pa.float64(), nullable=False),
            pa.field("y", pa.float64(), nullable=False),
        ]
    )
    fn = DuckDBInferFn(
        "SELECT fmod(x, y) AS f, x % y AS m FROM __THIS__",
        row_tables={"__THIS__": schema},
        static_tables={},
    )
    (got,) = fn.infer_rows([{"x": 7.5, "y": 0.0}])
    assert bits(got["f"]) == "fff8000000000000"

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (x DOUBLE, y DOUBLE)")
    con.execute("INSERT INTO __THIS__ VALUES (7.5, 0.0)")
    f, m = con.execute("SELECT fmod(x, y), x % y FROM __THIS__").fetchone()
    assert bits(f) == "fff8000000000000"
    assert bits(got["m"]) == bits(m), f"{bits(got['m'])} vs oracle {bits(m)}"


# ----------------------------------------------------------- nextafter:
# C nextafter bit-exact, TOTAL (no traps); x == y returns y including the
# signed-zero identity; repr equality in duck_check pins every bit pattern.


def test_nextafter_bit_exact_witnesses():
    duck_check(
        "SELECT nextafter(x, y) AS n FROM __THIS__",
        {"x": "float?", "y": "float?"},
        [
            {"x": 1.0, "y": 2.0},  # 1.0000000000000002
            {"x": 1.0, "y": 0.0},  # 0.9999999999999999
            {"x": 0.0, "y": -1.0},  # -5e-324
            {"x": -0.0, "y": 1.0},  # 5e-324
            {"x": 0.0, "y": -0.0},  # -0.0 (x == y returns y; repr pins sign)
            {"x": -0.0, "y": 0.0},  # 0.0
            {"x": 1.7976931348623157e308, "y": INF},  # inf
            {"x": INF, "y": 0.0},  # 1.7976931348623157e308
            {"x": -INF, "y": 0.0},  # -1.7976931348623157e308
            {"x": INF, "y": INF},  # inf (x == y)
            {"x": NAN, "y": 1.0},  # nan
            {"x": 1.0, "y": NAN},  # nan
            {"x": None, "y": 1.0},  # NULL-strict
            {"x": 1.0, "y": None},
            {"x": None, "y": None},
        ],
    )


def test_nextafter_int_args_promote_to_double():
    duck_check(
        "SELECT nextafter(a, b) AS n FROM __THIS__",
        {"a": "int", "b": "int"},
        [{"a": 1, "b": 2}, {"a": 0, "b": -1}],  # 1.0000000000000002 / -5e-324
    )
