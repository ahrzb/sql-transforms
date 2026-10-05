"""UBIGINT and HUGEINT on the i128 lane.

HUGEINT is a lane of its own (checked i128 arithmetic, exported as
decimal128(38, 0) the way DuckDB exports it); UBIGINT is its narrow width,
computed on i128 and range-checked to [0, 2^64), as UTINYINT..UINTEGER are
on the i64 lane. Typing is DuckDB 1.5.5's, measured with the optimizer off:

- an operator over UBIGINT and a signed width is HUGEINT unless the signed
  side is strictly wider (`u64 + i64`, `u64 & i8` are HUGEINT), over UBIGINT
  and an unsigned width UBIGINT; HUGEINT with any integer is HUGEINT;
- CASE / COALESCE / greatest / least: the smallest signed type holding both
  (`coalesce(u64, i8)` is HUGEINT);
- a literal that fits a non-literal side takes its type (`u64 + 1` and
  `u64 + 9223372036854775808` are UBIGINT; `u64 + (-1)` is HUGEINT);
- 9223372036854775808 is a HUGEINT literal, `-9223372036854775808` BIGINT,
  `- -9223372036854775808` HUGEINT again, past HUGEINT a literal is
  UHUGEINT (refused) and past that DOUBLE;
- a DECIMAL meets UBIGINT as DECIMAL(20, 0) and HUGEINT as DECIMAL(38, 0);
- round(UBIGINT) is HUGEINT, floor/ceil/sqrt are DOUBLE, abs keeps the width.

What DuckDB computes in a way this engine does not reproduce refuses by
name: unary minus over UBIGINT (it wraps), shifts over either, round/trunc
with digits, and UHUGEINT.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pytest
from confit import DuckDBInferFn

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity  # noqa: E402

U64_MAX = 2**64 - 1
I128_MAX = 2**127 - 1

ROWS = pa.table(
    {
        "u64": pa.array([U64_MAX, None, 0, 2**63], pa.uint64()),
        "u8": pa.array([200, None, 0, 255], pa.uint8()),
        "u32": pa.array([4000000000, 7, 0, None], pa.uint32()),
        "i8": pa.array([-5, 3, None, 127], pa.int8()),
        "i64": pa.array([-5, 10, None, 2**63 - 1], pa.int64()),
        "f": pa.array([2.5, -0.5, None, 1.8446744073709552e19], pa.float64()),
        "s": pa.array(["17", "-0", None, "18446744073709551615"]),
        "d": pa.array(
            [Decimal("1.5"), Decimal("-2.25"), None, Decimal("7")], pa.decimal128(9, 4)
        ),
    }
)
SMALL = ROWS.slice(1, 2)
H = "CAST(i64 AS HUGEINT)"


@pytest.mark.parametrize(
    "expr",
    [
        "u64",
        "u64 + 0",
        "u64 // 2",
        "u64 % 7",
        "u64 // 0",
        "u64 % 0",
        "u64 + i64",
        "u64 + i8",
        "u64 - u8",
        "u64 + u32",
        "u64 * 0",
        "u64 + (-1)",
        "u64 / 3",
        "u64 & 255",
        "u64 | i8",
        "xor(u64, u32)",
        "u64 > i64",
        "u64 = -1",
        "u64 = 18446744073709551615",
        "u64 IN (1, 0, -1)",
        "u64 BETWEEN -1 AND 5",
        "u64 = 2.5",
        "u64 < f",
        "u64 = s",
        "coalesce(u64, i64)",
        "coalesce(u64, u32)",
        "coalesce(u64, 1)",
        "coalesce(u64, 9223372036854775808)",
        "greatest(u64, i8)",
        "least(u64, 1)",
        "nullif(u64, 0)",
        "CASE WHEN i8 > 0 THEN u64 ELSE i64 END",
        "CASE WHEN i8 > 0 THEN u64 ELSE 1 END",
        "abs(u64)",
        "round(u64)",
        "trunc(u64)",
        "sqrt(u64)",
        "floor(u64)",
        "u64 || 'x'",
        "u64 + 1.5",
        "u64 * d",
        "u64 = d",
        "CAST(u64 AS VARCHAR)",
        "CAST(u64 AS DOUBLE)",
        "CAST(u64 AS BOOLEAN)",
        "CAST(u64 AS DECIMAL(20,0))",
        "CAST(u64 AS HUGEINT) * 3",
        "TRY_CAST(u64 AS BIGINT)",
        "TRY_CAST(u64 AS INTEGER)",
        "TRY_CAST(u64 AS DECIMAL(19,0))",
        "TRY_CAST(i64 AS UBIGINT)",
        "TRY_CAST(s AS UBIGINT)",
        "TRY_CAST(s AS HUGEINT)",
        "TRY_CAST(s AS UINT64)",
        "TRY_CAST(f AS UBIGINT)",
        "TRY_CAST(f AS HUGEINT)",
        "TRY_CAST(d AS UBIGINT)",
        f"TRY_CAST({H} AS UBIGINT)",
        f"TRY_CAST({H} * {H} AS UBIGINT)",
        "CAST(i8 > 0 AS HUGEINT)",
        "CAST(i8 AS INT128) * 2",
        # HUGEINT
        f"{H} * 170141183460469231731687303715884105727 // {H}",
        f"{H} + 2.5",
        f"{H} = d",
        f"-{H}",
        f"abs({H})",
        f"round({H})",
        f"ceil({H})",
        f"{H} % 0",
        f"{H} // 3",
        f"{H} % -3",
        f"CAST({H} AS DOUBLE)",
        f"CAST({H} AS VARCHAR) || 'x'",
        "9223372036854775808",
        "9223372036854775808 + i64",
        "-9223372036854775808",
        "-9223372036854775809",
        "- -9223372036854775808",
        "-(-9223372036854775809)",
        "170141183460469231731687303715884105727 - i8",
        "-170141183460469231731687303715884105728",
        "340282366920938463463374607431768211456",
        "u64 + 9223372036854775808",
        "i8 * 9223372036854775808",
        "TRY_CAST(9223372036854775808 AS BIGINT)",
        # VARCHAR into the narrow unsigned widths: the sign rule (slice 1
        # parsed '-0.4' as 0 where DuckDB fails the cast)
        "TRY_CAST(' -0.4' AS UTINYINT)",
        "TRY_CAST(s AS UTINYINT)",
    ],
)
def test_a_wide_integer_expression_agrees_with_the_oracle(expr):
    assert_parity(f"SELECT {expr} AS o FROM __THIS__", ROWS)


@pytest.mark.parametrize(
    "expr",
    [
        "u64 - 1",
        "u64 * 2",
        "u64 + u8",
        "u64 + 18446744073709551615",
        "CAST(u64 AS BIGINT)",
        "CAST(i64 AS UBIGINT)",
        "CAST(s AS UBIGINT) + 1",
        "CAST(f AS UBIGINT)",
        "CAST(d AS UBIGINT)",
        f"CAST({H} AS UBIGINT)",
        f"{H} * 170141183460469231731687303715884105727",
        "170141183460469231731687303715884105727 + i8",
        f"abs({H} - 170141183460469231731687303715884105727 - 2)",
        f"CAST({H} * 100000000000000000000000000000000 AS DECIMAL(38,1))",
    ],
)
def test_an_overflow_traps_on_both_engines(expr):
    assert_parity(f"SELECT {expr} AS o FROM __THIS__", ROWS, expect="AGREE_TRAP")


@pytest.mark.parametrize(
    "expr, why",
    [
        ("-u64", "unary minus"),
        ("u64 << 1", "shift"),
        (f"{H} >> 1", "shift"),
        ("round(u64, -1)", "digits"),
        (f"trunc({H}, -1)", "digits"),
        ("170141183460469231731687303715884105728", "UHUGEINT"),
        ("CAST(i64 AS UHUGEINT)", "UHUGEINT"),
    ],
)
def test_what_duckdb_computes_differently_refuses_by_name(expr, why):
    v = assert_parity(f"SELECT {expr} AS o FROM __THIS__", SMALL, expect="REFUSED")
    assert why in v.detail


@pytest.mark.parametrize(
    "fn",
    [
        "lpad('x', u64, 'y')",
        "repeat('x', u64)",
        "substr('abc', u64)",
        # the NULL short-circuit does not skip the count's type (campaign
        # seed 12693)
        "lpad(NULL, 18446744073709551616, 'y')",
        "lpad(NULL, u64, 'y')",
    ],
)
def test_a_count_argument_does_not_narrow_on_either_engine(fn):
    # DuckDB has no implicit UBIGINT -> BIGINT/INTEGER: a binder error there.
    assert_parity(f"SELECT {fn} AS o FROM __THIS__", SMALL, expect="REFUSED")


def test_a_ubigint_static_column_joins_and_reads():
    s = pa.table(
        {
            "k": pa.array([U64_MAX, 7, 2**63], pa.uint64()),
            "v": pa.array([1, None, U64_MAX], pa.uint64()),
        }
    )
    assert_parity(
        "SELECT u64, s.k, s.v + 0 AS w FROM __THIS__ LEFT JOIN s ON u64 = s.k",
        ROWS,
        statics={"s": s},
    )


@pytest.mark.parametrize(
    "on, static",
    [
        # a signed probe against a UBIGINT build key compares as HUGEINT
        ("i64 = s.k", pa.array([10, 2**63 - 1, U64_MAX], pa.uint64())),
        # a UBIGINT probe against a signed build key
        ("u64 = s.k", pa.array([0, -1, 2**63 - 1], pa.int64())),
        ("u64 = s.k", pa.array(["18446744073709551615", "0", " 7"])),
        (
            "CAST(u64 AS DOUBLE) = s.k",
            pa.array([1.8446744073709552e19, 0.0], pa.float64()),
        ),
    ],
)
def test_a_mixed_width_join_key_matches_duckdbs(on, static):
    s = pa.table({"k": static})
    # INNER: the matches decide the rows (a VARCHAR build key cannot be
    # projected back through a numeric comparison, a refusal of its own).
    assert_parity(
        f"SELECT u64, i64 FROM __THIS__ JOIN s ON {on}", ROWS, statics={"s": s}
    )


def test_both_boundaries_keep_the_types():
    fn = DuckDBInferFn(
        "SELECT u64, u64 // 2 AS h, CAST(i64 AS HUGEINT) * 4 AS g FROM __THIS__",
        row_tables={"__THIS__": ROWS.schema},
        static_tables={},
    )
    got = fn.infer_arrow(ROWS)
    assert got.schema.types == [pa.uint64(), pa.uint64(), pa.decimal128(38, 0)]
    assert got.column("u64").to_pylist() == [U64_MAX, None, 0, 2**63]
    rows = fn.infer_rows(ROWS.to_pylist())
    assert rows == got.to_pylist()


def test_a_hugeint_past_38_digits_crosses_both_boundaries():
    # DuckDB writes the i128 into decimal128(38, 0) whatever its digits.
    fn = DuckDBInferFn(
        "SELECT 170141183460469231731687303715884105727 + i8 AS o FROM __THIS__",
        row_tables={"__THIS__": ROWS.schema},
        static_tables={},
    )
    got = fn.infer_arrow(ROWS.slice(0, 1))
    assert got.to_pylist() == [{"o": Decimal(I128_MAX - 5)}]
    assert fn.infer_rows(ROWS.slice(0, 1).to_pylist()) == got.to_pylist()


def test_a_ubigint_row_value_outside_its_range_refuses_at_the_boundary():
    fn = DuckDBInferFn(
        "SELECT u64 FROM __THIS__",
        row_tables={"__THIS__": pa.schema([pa.field("u64", pa.uint64())])},
        static_tables={},
    )
    with pytest.raises(ValueError, match="outside its uint64 range"):
        fn.infer_rows([{"u64": -1}])
    assert fn.infer_rows([{"u64": U64_MAX}]) == [{"u64": U64_MAX}]


@pytest.mark.parametrize("cast", ["CAST", "TRY_CAST"])
def test_uhugeint_still_refuses_by_name(cast):
    with pytest.raises(ValueError, match="UHUGEINT"):
        DuckDBInferFn(
            f"SELECT {cast}(k AS UHUGEINT) AS o0 FROM __THIS__",
            row_tables={"__THIS__": pa.schema([pa.field("k", pa.int64())])},
            static_tables={},
        )


def test_a_ubigint_shared_column_joins_by_name():
    # Once an opaque shared column (tests/test_join_keys.py); a key now.
    s = pa.table(
        {"u64": pa.array([2**63, 0], pa.uint64()), "z": pa.array([1, 2], pa.int64())}
    )
    assert_parity(
        "SELECT u64, z FROM __THIS__ JOIN s USING (u64)", ROWS, statics={"s": s}
    )
    assert_parity("SELECT u64, z FROM __THIS__ NATURAL JOIN s", ROWS, statics={"s": s})


@pytest.mark.parametrize(
    "spelling", ["UBIGINT", "UINT64", "HUGEINT", "INT128", "UINT8", "UINT16", "UINT32"]
)
@pytest.mark.parametrize("cast", ["CAST", "TRY_CAST"])
def test_an_unaliased_cast_is_named_as_duckdb_names_it(cast, spelling):
    # An alias prints as the type it binds to: `CAST(i AS "UBIGINT")`.
    assert_parity(f"SELECT {cast}(i8 AS {spelling}) FROM __THIS__", SMALL)


@pytest.mark.parametrize("bad", ["-1", "18446744073709551616", "-0.4"])
def test_a_varchar_build_key_parses_at_the_unsigned_probe_type(bad):
    # DuckDB casts the VARCHAR key to UBIGINT: '-1' and '-0.4' fail its sign
    # rule and 2^64 its range, an error on every query. Parsed at the i128
    # lane, all three used to serve (review of #349).
    s = pa.table({"k": pa.array([bad, "3"]), "z": pa.array([1, 2], pa.int64())})
    v = assert_parity(
        "SELECT u64, z FROM __THIS__ JOIN s ON u64 = s.k",
        ROWS,
        statics={"s": s},
        expect="REFUSED",
    )
    assert "UBIGINT" in v.detail, v.detail


def test_a_varchar_build_key_in_range_still_joins_an_unsigned_probe():
    s = pa.table(
        {"k": pa.array(["18446744073709551615", "-0", " 7 "]), "z": pa.array([1, 2, 3])}
    )
    assert_parity(
        "SELECT u64, z FROM __THIS__ JOIN s ON u64 = s.k", ROWS, statics={"s": s}
    )


@pytest.mark.parametrize(
    "expr",
    [
        "u64 IN ('-1', 3)",
        "u64 IN ('-0.4', 1)",
        "u64 BETWEEN '-0.4' AND 5",
        f"{H} IN ('9007199254740993')",
    ],
)
def test_a_string_member_beside_a_wide_integer_refuses_by_name(expr):
    # The family's string conversion rounds through f64 to BIGINT: no sign
    # rule, no 128-bit precision (review of #349).
    v = assert_parity(f"SELECT {expr} AS o FROM __THIS__", ROWS, expect="REFUSED")
    assert "UBIGINT or HUGEINT" in v.detail
