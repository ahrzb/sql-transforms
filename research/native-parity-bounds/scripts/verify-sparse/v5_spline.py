import warnings; warnings.simplefilter("ignore")
import numpy as np, itertools
from sklearn.preprocessing import SplineTransformer
rng = np.random.default_rng(5)
X = rng.uniform(-3, 4, size=(80, 2))
P = np.vstack([X[:30], [[10.0, 10.0], [-10.0, -10.0], [4.5, 0.0], [0.0, 4.5], [-0.0, 0.0]],
               np.column_stack([np.nextafter(X.max(0)[0], np.inf), X.max(0)[1]])])
def bits(a): return np.ascontiguousarray(a, float).view(np.uint64)
summary = {}
for deg, nk, ext, bias, knots in itertools.product(range(0, 5), (2, 3, 5), ("constant", "linear", "continue", "periodic", "error"), (True, False), ("uniform", "quantile")):
    kw = dict(degree=deg, n_knots=nk, extrapolation=ext, include_bias=bias, knots=knots)
    try:
        s = SplineTransformer(sparse_output=True, **kw).fit(X); d = SplineTransformer(sparse_output=False, **kw).fit(X)
    except Exception as e:
        continue
    for r in P:
        rs = rd = None
        try: rs = s.transform([r]).toarray()
        except Exception as e: rs = type(e).__name__
        try: rd = d.transform([r])
        except Exception as e: rd = type(e).__name__
        if isinstance(rs, str) or isinstance(rd, str):
            key = (rs if isinstance(rs, str) else "ok", rd if isinstance(rd, str) else "ok")
            if key[0] != key[1]:
                summary.setdefault(("raise-asym", deg, ext, key), 0); summary[("raise-asym", deg, ext, key)] += 1
            continue
        if rs.shape != rd.shape or not np.array_equal(bits(rs), bits(rd)):
            k = ("diff", deg, ext, nk)
            summary.setdefault(k, []).append((r.tolist(), rs.round(3).tolist(), rd.round(3).tolist()))
for k, v in summary.items():
    print(k, v if isinstance(v, int) else (len(v), v[0]))
