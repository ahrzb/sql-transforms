"""Nystroem(rbf): twin = est.transform([row])[0]; native = clipped expansion (row_sumsq XX, numpy BB), arg = q*-gamma,
glibc exp, ltr matvec over components. S_l = sum_j |N_lj| k_j (1 + gamma S2_j); S_mv = sum_j |N_lj| k_j."""
import math, numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.utils.extmath import row_norms
src = open("/home/user/sql-transforms/packages/sql-transform/sql_transform/native/_helpers.py").read()
s = src.index("def row_sumsq("); e = src.index("@functools.cache", s)
ns = {}; exec("from typing import Any\n" + src[s:e], ns); row_sumsq = ns["row_sumsq"]
EPS = 2.0 ** -52
rng = np.random.default_rng(2718)
allK = []
for n, m, off, g in ((6, 60, 0.0, None), (6, 60, 30.0, None), (40, 200, 0.0, 0.2), (40, 200, 50.0, None), (12, 120, 5.0, 3.0)):
    X = rng.normal(size=(1500, n)) * 2 + off
    est = Nystroem(n_components=m, gamma=g, random_state=1).fit(X)
    gamma = (1.0 / n) if g is None else g
    Bm = est.components_; Bl = [list(map(float, b)) for b in Bm]; BB = row_norms(Bm, squared=True); N = est.normalization_
    rows = np.vstack([X[rng.choice(1500, 150, replace=False)], Bm[rng.integers(0, m, 80)] * (1 + 1e-10 * rng.normal(size=(80, n)))])
    K, Kmv = [], []
    for r in rows:
        x = [float(v) for v in r]; tw = est.transform([x])[0]; xx = row_sumsq(x)
        q = []
        for j in range(m):
            a = x[0] * Bl[j][0]
            for i in range(1, n): a = a + x[i] * Bl[j][i]
            q.append(max((-2.0 * a + xx) + BB[j], 0.0))
        k = np.array([math.exp(v * -gamma) for v in q])
        y = k[0] * N[:, 0]
        for j in range(1, m): y = y + k[j] * N[:, j]
        S2 = np.array([math.fsum(v * v for v in x) + BB[j] + 2 * math.fsum(abs(p * c) for p, c in zip(x, Bl[j])) for j in range(m)])
        S = (np.abs(N) * (k * (1 + gamma * S2))[None, :]).sum(axis=1); Smv = (np.abs(N) * k[None, :]).sum(axis=1)
        d = np.abs(y - tw); K.append(d / (EPS * S)); Kmv.append(d / (EPS * Smv))
    K, Kmv = np.concatenate(K), np.concatenate(Kmv); allK.append(K)
    print(f"Nystroem n={n} m={m} off={off} gamma={gamma:.3g}: lanes={K.size} K max={K.max():.3f} | K(plain matvec scale) max={Kmv.max():.3g}", flush=True)
print("POOLED K max", np.concatenate(allK).max())
