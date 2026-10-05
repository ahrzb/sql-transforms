"""greatest/least: NULL-skipping, first maximal element on ties, every
argument evaluated -- and linear-ish build cost in the argument count (the
pairwise fold was 4^n)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pyarrow as pa
import pytest
from confit import DuckDBInferFn

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity  # noqa: E402

NAN = float("nan")
ROWS = pa.table(
    {
        "a": [1.0, None, NAN, -0.0, 2.5, None],
        "b": [0.0, None, 1.0, 0.0, None, 3.0],
        "c": [1.0, 2.0, None, -0.0, 2.5, None],
        "k": [3, None, -1, 0, 7, 2],
        "s": ["b", None, "a", "", "zz", "b"],
        "t": ["a", "x", None, "", "zz", None],
    }
)


@pytest.mark.parametrize("fn", ["greatest", "least"])
@pytest.mark.parametrize(
    "args",
    [
        "a, b",
        "a, b, c",
        "c, b, a",
        "a, NULL, b",
        "a, k",
        "k, a, b, c, k * 2",
        "s, t",
        "t, s, 'm'",
        "a",
        "a * 1.5, b - 1, c + 0.5, a, b, c, k, -a",
    ],
)
def test_greatest_least_agree_with_the_oracle(fn, args):
    assert_parity(f"SELECT {fn}({args}) AS o FROM __THIS__", ROWS)


def test_a_trapping_argument_still_traps():
    assert_parity(
        "SELECT greatest(a, k + 9223372036854775807) AS o FROM __THIS__",
        ROWS,
        trap="Overflow",
    )


def test_many_arguments_build_quickly():
    sch = pa.schema([(f"x{i}", pa.float64()) for i in range(16)])
    t = time.perf_counter()
    DuckDBInferFn(
        f"SELECT least({', '.join(f'x{i} * 2' for i in range(16))}) AS o FROM __THIS__",
        row_tables={"__THIS__": sch},
        static_tables={},
    )
    assert time.perf_counter() - t < 5


INTS = pa.table(
    {
        "i": pa.array([1, None, -128, 127, 0], pa.int8()),
        "j": pa.array([1, 5, None, -5, 0], pa.int32()),
        "h": pa.array([2**63 - 1, None, -(2**63), 0, None], pa.int64()),
        "z": pa.array([-0.0, 0.0, None, float("nan"), float("-inf")], pa.float64()),
        "w": pa.array([0.0, -0.0, float("nan"), float("nan"), None], pa.float64()),
    }
)


@pytest.mark.parametrize("fn", ["greatest", "least"])
@pytest.mark.parametrize(
    "args",
    [
        "i, j",
        "j, i, h",
        "i, i, i",
        "h, j",
        "z, w",
        "w, z, 0.0",
        "i, j, z",
        "NULL, i, NULL",
    ],
)
def test_the_running_extreme_agrees_on_widths_ties_and_nan(fn, args):
    # Lowered as a running extreme (lower.rs, SKind::Extreme): the first
    # maximal argument wins a tie (-0.0 before 0.0), NaN orders above +inf,
    # NULLs are skipped, and the result keeps the unified width.
    assert_parity(f"SELECT {fn}({args}) AS o FROM __THIS__", INTS)


def test_the_first_trapping_argument_traps_first():
    # Every argument is evaluated in order, so the first one that traps
    # names the error, as on DuckDB.
    assert_parity(
        "SELECT greatest(j, CAST(h AS INTEGER), h + 1) AS o FROM __THIS__",
        INTS,
        trap="(?i)out of range|overflow|conver",
    )


def test_hundreds_of_arguments_build_linearly():
    n = 512
    sch = pa.schema([(f"x{i}", pa.float64()) for i in range(n)])
    args = ", ".join(f"abs(x{i})" for i in range(n))
    t = time.perf_counter()
    DuckDBInferFn(
        f"SELECT greatest({args}) AS o FROM __THIS__",
        row_tables={"__THIS__": sch},
        static_tables={},
    )
    # 0.05 s on a release build (the flat CASE refused past 128 arguments).
    assert time.perf_counter() - t < 20
