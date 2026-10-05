"""UTINYINT / USMALLINT / UINTEGER: narrow widths of the i64 lane.

Columns (row and static), CAST targets, and their typing as DuckDB 1.5.5
does it (measured): an operator over unsigned and unsigned is the wider
unsigned, over signed and unsigned the signed side when strictly wider and
BIGINT otherwise; CASE / COALESCE / greatest take the smallest signed type
holding both; a literal that fits an unsigned side takes its type. Overflow
traps at the unsigned width. What DuckDB computes differently -- unary
minus wraps, a DOUBLE is range-checked before rounding, shifts trap at the
width -- refuses by name. UBIGINT and HUGEINT ride the i128 lane
(tests/test_hugeint.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest
from confit import DuckDBInferFn

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity  # noqa: E402

ROWS = pa.table(
    {
        "u8": pa.array([200, None, 0, 255], pa.uint8()),
        "u16": pa.array([60000, 1, None, 65535], pa.uint16()),
        "u32": pa.array([4000000000, 7, 0, None], pa.uint32()),
        "i8": pa.array([-5, 3, None, 127], pa.int8()),
        "i16": pa.array([-5, 300, 2, None], pa.int16()),
        "i32": pa.array([-5, None, 9, 1], pa.int32()),
        "i64": pa.array([-5, 10, None, 2], pa.int64()),
        "s": pa.array(["17", "300", None, "-0"]),
    }
)
SMALL = ROWS.slice(1, 2)


@pytest.mark.parametrize(
    "expr",
    [
        "u8",
        "u16 + 0",
        "u8 + i8",
        "u8 + i16",
        "u16 + i16",
        "u32 + i32",
        "u32 + i64",
        "u8 + 300",
        "u8 - u8",
        "u8 // 3",
        "u8 % 7",
        "u8 // 0",
        "u8 / 3",
        "u8 & 3",
        "u8 | i8",
        "xor(u8, u16)",
        "u8 > i8",
        "u32 > 4294967294",
        "greatest(u8, i8)",
        "least(u8, 1)",
        "coalesce(u8, i8)",
        "coalesce(u16, i8)",
        "coalesce(u32, i16)",
        "CASE WHEN i8 > 0 THEN u8 ELSE i8 END",
        "TRY_CAST(i8 AS UTINYINT)",
        "TRY_CAST(u32 AS INTEGER)",
        "CAST(u8 AS VARCHAR)",
        "TRY_CAST(s AS UINT8)",
        "CAST(2.5 AS UTINYINT)",
        "abs(u8)",
        "round(u8)",
        "round(u16, 1)",
        "trunc(u16, 1)",
        "u8 + 0.5",
        "u8 IN (300, 200)",
        "u8 || 'x'",
        "nullif(u8, 200)",
        "CAST(u8 AS DECIMAL(4,1))",
        "u32 + 5000000000",
        "CAST(4000000000 AS UINTEGER) + 1",
        "lpad('x', u8 % 5, 'y')",
    ],
)
def test_an_unsigned_expression_agrees_with_the_oracle(expr):
    assert_parity(f"SELECT {expr} AS o FROM __THIS__", ROWS)


@pytest.mark.parametrize(
    "expr",
    [
        "u8 + u16",
        "u8 + 1",
        "u8 - 201",
        "u16 * u16",
        "u32 + u32",
        "CAST(i8 AS UTINYINT)",
        "CAST(u32 AS INTEGER)",
        "CAST(i64 AS UINTEGER)",
        "CAST(s AS UTINYINT)",
    ],
)
def test_an_unsigned_overflow_traps_on_both_engines(expr):
    assert_parity(f"SELECT {expr} AS o FROM __THIS__", ROWS, expect="AGREE_TRAP")


@pytest.mark.parametrize(
    "expr, why",
    [
        ("-u8", "unary minus"),
        ("u8 << 1", "shift"),
        ("CAST(i64 * 1.5e0 AS UTINYINT)", "before rounding"),
        ("lpad('x', u32, 'y')", "UINTEGER"),
    ],
)
def test_what_duckdb_computes_differently_refuses_by_name(expr, why):
    v = assert_parity(f"SELECT {expr} AS o FROM __THIS__", SMALL, expect="REFUSED")
    assert why in v.detail


def test_an_unsigned_static_column_joins_and_reads():
    s = pa.table(
        {"k": pa.array([7, 1], pa.uint32()), "v": pa.array([250, 3], pa.uint8())}
    )
    assert_parity(
        "SELECT i64, s.v + 1 AS w FROM __THIS__ LEFT JOIN s ON u32 = s.k",
        ROWS,
        statics={"s": s},
    )


def test_both_boundaries_keep_the_unsigned_type():
    fn = DuckDBInferFn(
        "SELECT u8, u16 // 2 AS h, u32 FROM __THIS__",
        row_tables={"__THIS__": ROWS.schema},
        static_tables={},
    )
    got = fn.infer_arrow(ROWS)
    assert got.schema.types == [pa.uint8(), pa.uint16(), pa.uint32()]
    rows = fn.infer_rows(ROWS.to_pylist())
    assert rows == got.to_pylist()


@pytest.mark.parametrize("target", ["UHUGEINT", "UINT128"])
@pytest.mark.parametrize("cast", ["CAST", "TRY_CAST"])
def test_a_cast_to_an_unserved_width_refuses_by_name(cast, target):
    # Issue #220: these once typed as BIGINT and served DuckDB's refusals.
    with pytest.raises(ValueError, match=target):
        DuckDBInferFn(
            f"SELECT {cast}(k AS {target}) AS o0 FROM __THIS__",
            row_tables={"__THIS__": pa.schema([pa.field("k", pa.int64())])},
            static_tables={},
        )


def test_a_varchar_build_key_takes_the_unsigned_sign_rule():
    # '-0.4' is no UTINYINT on DuckDB (a minus only before zeros), so the
    # key fails to convert there on every query (review of #349).
    s = pa.table({"k": pa.array(["-0.4", "200"]), "z": pa.array([1, 2], pa.int64())})
    assert_parity(
        "SELECT u8, z FROM __THIS__ JOIN s ON u8 = s.k",
        ROWS,
        statics={"s": s},
        expect="REFUSED",
    )
    ok = pa.table({"k": pa.array(["-0", "200"]), "z": pa.array([1, 2], pa.int64())})
    assert_parity(
        "SELECT u8, z FROM __THIS__ JOIN s ON u8 = s.k", ROWS, statics={"s": ok}
    )
