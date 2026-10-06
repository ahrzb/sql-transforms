import sys; sys.path.insert(0, sys.argv[1]); import shim  # noqa
import warnings; warnings.simplefilter("ignore")
import time, numpy as np, scipy.sparse as sp, pyarrow as pa
from sklearn.preprocessing import OneHotEncoder
from sql_transform import PythonTransform
def t(f, n):
    best = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n): f()
        best.append((time.perf_counter() - t0) / n)
    return np.median(best) * 1e6
for k in (10, 1000, 100_000):
    cats = np.array([f"c{i}" for i in range(k)], dtype=object)[:, None]
    s = OneHotEncoder(handle_unknown="ignore").fit(cats); d = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cats)
    row = [["c3"]]
    n = 200 if k < 100_000 else 20
    out = s.transform(row)
    ts = t(lambda: s.transform(row), n); td = t(lambda: d.transform(row), n); ta = t(lambda: out.toarray(), n * 5)
    tf = t(lambda: tuple(float(v) for v in d.transform(row)[0]), n)
    print(f"k={k}: sparse transform {ts:.1f}us dense transform {td:.1f}us toarray {ta:.2f}us  dense transform+float-tuple {tf:.1f}us  share {ta/ts*100:.2f}%")
# return_types per call and post_init
for k in (1000, 10_000):
    names = [f"f{i}" for i in range(k)]
    ret = pa.struct([(n, pa.float64()) for n in names])
    t0 = time.perf_counter()
    pt = PythonTransform(name="t", instances={}, takes=pa.schema([("a", pa.float64())]), returns=ret)
    t1 = time.perf_counter()
    rt = t(lambda: pt.return_types, 5)
    print(f"k={k}: __post_init__ {t1-t0:.3f}s  return_types {rt/1e3:.2f} ms/call = {rt/k:.3f} us/lane")
