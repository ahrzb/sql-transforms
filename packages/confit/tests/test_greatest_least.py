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
