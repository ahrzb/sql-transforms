"""DECIMAL row columns: read from Arrow (every storage tier DuckDB exports)
and from Python rows, computed on, and emitted exactly.

Each query runs through the campaign verdict (both backends, against the
DuckDB oracle); the boundary checks are confit's own: a Python value must be
exactly representable at the column's (p, s).
"""

from __future__ import annotations

import decimal
import sys
from pathlib import Path

import pyarrow as pa
import pytest
from confit import DuckDBInferFn

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity  # noqa: E402

D = decimal.Decimal

ROWS = pa.table(
    {
        "d": pa.array(
            [D("1.5000"), None, D("-12345.9999"), D("0.0000")], pa.decimal128(9, 4)
        ),
        "a": pa.array([1, 2, None, 4], pa.int64()),
        "e": pa.array([D("3"), D("-7"), None, D("0")], pa.decimal128(38, 0)),
        "f": pa.array([D("0.01"), D("-0.99"), D("99.99"), None], pa.decimal128(4, 2)),
    }
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM __THIS__",
        "SELECT d, e, f FROM __THIS__",
        "SELECT d + 1 AS o, d * a AS p, d - f AS q FROM __THIS__",
        "SELECT d / 3 AS o, CAST(d AS DOUBLE) AS x, round(d, 1) AS r FROM __THIS__",
        "SELECT e + d AS o, e * e AS q, -e AS n FROM __THIS__",
        "SELECT CAST(d AS VARCHAR) AS s, d = 1.5 AS b, abs(d) AS ab FROM __THIS__",
        "SELECT a FROM __THIS__ WHERE d > 1 OR f < 0",
        "SELECT CASE WHEN d IS NULL THEN f ELSE d END AS o FROM __THIS__",
        "SELECT coalesce(d, 0) AS o, greatest(d, f) AS g FROM __THIS__",
        "SELECT o FROM (SELECT d * 2 AS o FROM __THIS__) AS t WHERE o < 100",
    ],
)
def test_a_decimal_row_column_agrees_with_the_oracle(sql):
    assert_parity(sql, ROWS)


def test_a_decimal_join_key_against_a_decimal_build_key_still_refuses():
    # PLANS, "Decimal remainders": DECIMAL probe keys are served against
    # DOUBLE build keys only.
    s = pa.table(
        {"k": pa.array([D("1.5000")], pa.decimal128(9, 4)), "v": pa.array(["x"])}
    )
    v = assert_parity(
        "SELECT a, s.v FROM __THIS__ LEFT JOIN s ON d = s.k",
        ROWS,
        statics={"s": s},
        expect="REFUSED",
    )
    assert "DECIMAL probe keys" in v.detail


@pytest.mark.parametrize(
    "tier", [pa.decimal32(9, 4), pa.decimal64(9, 4), pa.decimal128(9, 4)]
)
def test_every_storage_tier_reads(tier):
    fn = DuckDBInferFn(
        "SELECT d * 2 AS o FROM __THIS__",
        row_tables={"__THIS__": pa.schema([("d", pa.decimal128(9, 4))])},
        static_tables={},
    )
    t = pa.table({"d": pa.array([D("1.2500"), None], tier)})
    assert fn.infer_arrow(t).to_pylist() == [{"o": D("2.5000")}, {"o": None}]


def _rows_fn():
    return DuckDBInferFn(
        "SELECT d * 2 AS o FROM __THIS__",
        row_tables={"__THIS__": pa.schema([("d", pa.decimal128(9, 4))])},
        static_tables={},
    )


def test_python_rows_take_a_decimal_or_an_int():
    got = _rows_fn().infer_rows(
        [{"d": D("1.25")}, {"d": 2}, {"d": D("-0.1000")}, {"d": None}]
    )
    assert got == [
        {"o": D("2.5000")},
        {"o": D("4.0000")},
        {"o": D("-0.2000")},
        {"o": None},
    ]


@pytest.mark.parametrize(
    "value, why",
    [
        (D("1.25551"), "more than 4 decimal places"),
        (D("123456.0"), "out of range"),
        (D("NaN"), "not finite"),
        (1.5, "got float"),
        (True, "got bool"),
        ("1.5", "got str"),
    ],
)
def test_a_python_value_must_be_exact(value, why):
    with pytest.raises(ValueError, match=why):
        _rows_fn().infer_rows([{"d": value}])


def test_an_arrow_column_at_another_scale_refuses():
    t = pa.table({"d": pa.array([D("1.50")], pa.decimal128(9, 2))})
    with pytest.raises(ValueError, match="cast first"):
        _rows_fn().infer_arrow(t)
