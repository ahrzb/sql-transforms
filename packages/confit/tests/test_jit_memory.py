"""A dropped function returns its JIT memory.

cranelift_jit's JITModule frees its code and data only through
`free_memory`; dropping it kept 2 mappings per build, so a process that
rebuilt functions reached `vm.max_map_count` after about 32k builds and
stalled (reported by the native catalog loop, 2026-10-05).
"""

import sys

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, SqlFunction
from confit import sql as S

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads /proc/self/maps"
)

TAKES = pa.schema([pa.field("__iid", pa.int64()), pa.field("x", pa.float64())])
ROW = pa.table({"__iid": pa.array([0], pa.int64()), "x": pa.array([1.0])})


def _maps() -> int:
    with open("/proc/self/maps") as f:
        return sum(1 for _ in f)


def _build_call_drop(i: int) -> None:
    fn = SqlFunction(
        "tf",
        TAKES,
        pa.float64(),
        lambda iid, x: S.case(iid == S.lit(0), x * S.lit(float(i))).otherwise(
            S.lit(None, pa.float64())
        ),
    )
    f = DuckDBInferFn(
        "SELECT tf(__iid, x) AS o FROM __THIS__",
        row_tables={"__THIS__": TAKES},
        static_tables={},
        udfs=[fn],
    )
    assert f.infer_arrow(ROW).column("o").to_pylist() == [float(i)]


def test_builds_do_not_accumulate_mappings():
    for i in range(20):
        _build_call_drop(i)
    before = _maps()
    for i in range(300):
        _build_call_drop(i)
    # Leaking, 300 builds add 600 mappings.
    assert _maps() - before < 50
