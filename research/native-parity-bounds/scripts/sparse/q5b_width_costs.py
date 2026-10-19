"""Q5b: the step's pre-existing per-row width costs, for scale against the
densify: PythonTransform.return_types is recomputed on every __call__
(_udf.py:316), and __post_init__'s case-collision check is O(k^2)."""
import sys, timeit
sys.path.insert(0, sys.argv[1])
import _shim  # noqa
import pyarrow as pa
from sql_transform._udf import PythonTransform
for k in (10, 1000, 10_000, 100_000):
    ret = pa.struct([(f"f{i}", pa.float64()) for i in range(k)])
    st = object.__new__(PythonTransform)
    for a, v in (("name", "t"), ("instances", {}), ("takes", pa.schema([("x", pa.float64())])), ("returns", ret)):
        object.__setattr__(st, a, v)
    n = max(3, 20000 // k)
    t = min(timeit.repeat(lambda: len(st.return_types), number=n, repeat=5)) / n
    print(f"k={k:6d}: return_types per call {t*1e6:10.1f} us  ({t*1e9/k:.0f} ns/lane)")
