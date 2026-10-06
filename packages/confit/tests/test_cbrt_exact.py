"""`cbrt` is DuckDB's, bit for bit, on Linux.

DuckDB calls the C runtime's `cbrt`, which glibc does not round correctly;
confit's toolchain ships its own, which differed from it on about half of
100,001 draws by up to 3 ulps (reported by the native catalog loop). The
kernel now calls glibc's (`kernels::duck_cbrt`). The parity contract's
1-ulp `cbrt` allowance (claim: cbrt-ulp-tolerance) is for builds on other
platforms; this test compares bits.
"""

from __future__ import annotations

import sys

import numpy as np
import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from confit.oracle import Oracle

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="glibc's cbrt is the reference"
)


def test_cbrt_is_duckdbs_bit_for_bit():
    rng = np.random.default_rng(7)
    x = np.concatenate(
        [
            rng.uniform(-1e3, 1e3, 2000),
            np.exp(rng.uniform(-700, 700, 2000)) * rng.choice([-1.0, 1.0], 2000),
            [23.64324940051347, 27.0, -8.0, 0.0, -0.0, 5e-324, np.inf, -np.inf, np.nan],
        ]
    )
    t = pa.table({"x": x})
    got = (
        DuckDBInferFn(
            "SELECT cbrt(x) AS y FROM __THIS__",
            row_tables={"__THIS__": t.schema},
            static_tables={},
        )
        .infer_arrow(t)
        .column("y")
        .to_numpy()
    )
    o = Oracle().load("v", t)
    want = o.sql("SELECT cbrt(x) AS y FROM v").to_arrow_table().column("y").to_numpy()
    same = (got.view(np.int64) == want.view(np.int64)) | (
        np.isnan(got) & np.isnan(want)
    )
    assert same.all(), f"{(~same).sum()} differ, e.g. x={x[~same][:3]}"
