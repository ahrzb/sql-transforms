"""Q5: per-row cost of densifying, against the rest of the step's row path.
Times (median of repeats, microseconds per call) for one OneHotEncoder
feature with k categories:
  T_sp    est_sparse.transform([row])
  T_de    est_dense.transform([row])            (the sparse_output=False twin)
  T_toar  out_sparse.toarray()                  (the densify itself, 1 x k)
  T_full  densifying __call__ path: transform -> toarray()[0] -> float tuple
  T_dfull dense-config __call__ path: transform -> [0] -> float tuple
"""
import sys
import time
import timeit
builds = {}

sys.path.insert(0, sys.argv[1])
import _shim  # noqa: F401,E402

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder  # noqa: E402

from sql_transform._udf import PythonTransform, _flatten_row  # noqa: E402


def med(fn, n):
    ts = timeit.repeat(fn, number=n, repeat=5)
    return 1e6 * sorted(ts)[2] / n


class Densifying(PythonTransform):
    """The proposed change, outside the repo: densify before [0]."""

    def __call__(self, iid, *feats):
        est = self.instances[iid]
        out = est.transform([[float(f) for f in feats]])
        if sp.issparse(out):
            out = out.toarray()
        return tuple(float(v) for v in _flatten_row(out[0]))


print(f"{'k':>7s} {'T_sp':>8s} {'T_de':>8s} {'T_toar':>8s} {'T_full':>8s} {'T_dfull':>8s}  {'dens/total':>9s} bytes/row")
for k in (10, 100, 1_000, 10_000, 100_000):
    X = np.arange(k, dtype=float).reshape(-1, 1)
    es = OneHotEncoder().fit(X)
    ed = OneHotEncoder(sparse_output=False).fit(X)
    row = [[float(k // 2)]]
    n = max(3, int(2000 / (1 + k / 50)))
    t_sp = med(lambda: es.transform(row), n)
    t_de = med(lambda: ed.transform(row), n)
    o = es.transform(row)
    t_to = med(lambda: o.toarray(), n)
    ret = pa.struct([(f"f{i}", pa.float64()) for i in range(k)])
    t0 = time.perf_counter()
    if k <= 10_000:
        step = Densifying("t", {0: es}, pa.schema([("x", pa.float64())]), ret)
        t_build = time.perf_counter() - t0
    else:  # skip the O(k^2) __post_init__ name check, timed separately
        step = object.__new__(Densifying)
        for a, v in (("name", "t"), ("instances", {0: es}), ("takes", pa.schema([("x", pa.float64())])), ("returns", ret)):
            object.__setattr__(step, a, v)
        t_build = float("nan")
    stepd = object.__new__(PythonTransform)
    for a, v in (("name", "t"), ("instances", {0: ed}), ("takes", pa.schema([("x", pa.float64())])), ("returns", ret)):
        object.__setattr__(stepd, a, v)
    builds[k] = t_build
    assert step(0, float(k // 2)) == stepd(0, float(k // 2))
    t_full = med(lambda: step(0, float(k // 2)), n)
    t_dfull = med(lambda: stepd(0, float(k // 2)), n)
    print(f"{k:7d} {t_sp:8.1f} {t_de:8.1f} {t_to:8.2f} {t_full:8.1f} {t_dfull:8.1f}  {t_to / t_full:9.3f} {8 * k}")

print('PythonTransform(...) construction seconds by k:', {k: round(v, 3) for k, v in builds.items()})
# KBins onehot: the record's second default
X = np.random.default_rng(0).normal(size=(500, 4))
ks = KBinsDiscretizer(encode="onehot", quantile_method="averaged_inverted_cdf").fit(X)
kd = KBinsDiscretizer(encode="onehot-dense", quantile_method="averaged_inverted_cdf").fit(X)
row = [X[0].tolist()]
n = 3000
o = ks.transform(row)
print(f"\nKBins(onehot, 4 feats x 5 bins): T_sp {med(lambda: ks.transform(row), n):.1f}"
      f"  T_de {med(lambda: kd.transform(row), n):.1f}  T_toar {med(lambda: o.toarray(), n):.2f} us")
