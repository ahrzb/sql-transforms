"""Distance K2 vs n with positive (uncentred, same-sign) data and with centred mixed-sign data."""
import math
from fractions import Fraction
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.extmath import row_norms
src = open("/home/user/sql-transforms/packages/sql-transform/sql_transform/native/_helpers.py").read()
s = src.index("def row_sumsq("); e = src.index("@functools.cache", s)
ns = {}; exec("from typing import Any\n" + src[s:e], ns); row_sumsq = ns["row_sumsq"]
EPS = 2.0 ** -52
rng = np.random.default_rng(64)
for kind in ("positive", "centred"):
    for n in (8, 32, 128, 512):
        X = rng.uniform(0, 1, size=(1200, n)) + rng.integers(0, 4, size=(1200, 1)) if kind == "positive" else rng.normal(size=(1200, n)) + rng.normal(size=(1, n)) * 0
        km = KMeans(n_clusters=6, n_init=1, random_state=0, max_iter=30).fit(X); C = km.cluster_centers_
        Cl = [list(map(float, c)) for c in C]; CC = row_norms(C, squared=True)
        nr = max(60, 40000 // (6 * n))
        rows = np.vstack([X[rng.choice(1200, nr // 2, replace=False)], C[rng.integers(0, 6, nr // 2)] * (1 + 1e-7 * rng.normal(size=(nr // 2, n)))])
        K2 = []
        for r in rows:
            x = [float(v) for v in r]; tw = km.transform([x])[0]; xx = row_sumsq(x)
            for k in range(6):
                a = x[0] * Cl[k][0]
                for i in range(1, n): a = a + x[i] * Cl[k][i]
                dn = math.sqrt(max((-2.0 * a + xx) + CC[k], 0.0)); dt = float(tw[k])
                S2 = math.fsum(v * v for v in x) + CC[k] + 2 * math.fsum(abs(p * q) for p, q in zip(x, Cl[k]))
                K2.append(abs(float(Fraction(dn) ** 2 - Fraction(dt) ** 2)) / (EPS * S2))
        K2 = np.array(K2)
        print(f"{kind:<9} n={n:>4} lanes={K2.size:>6} K2 max={K2.max():.3f} p99.9={np.percentile(K2,99.9):.3f}  (derived n+4={n+4})", flush=True)
