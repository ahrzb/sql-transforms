"""Independent re-measurement of the distance family. Different data/seeds from exp1.
twin  = est.transform([row])[0]
native= sqrt(max(((-2*dot_ltr) + XX) + CC, 0)) with XX from the repo's row_sumsq (imported
        from the repo file itself, executed standalone), CC = row_norms(C) constants.
K2 = |dn^2 - dt^2| / (eps S2) exactly; K1 = |dn-dt|/(eps sqrt S2); also vs derived (n+4)."""
import math, sys, importlib.util, types
from fractions import Fraction
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans, Birch
from sklearn.utils.extmath import row_norms

# load row_sumsq from the repo source text (avoid importing the package, which fails on py3.14rc2)
src = open("/home/user/sql-transforms/packages/sql-transform/sql_transform/native/_helpers.py").read()
start = src.index("def row_sumsq(")
end = src.index("@functools.cache", start)
ns = {"Any": object}
exec("from typing import Any\n" + src[start:end], ns)
row_sumsq = ns["row_sumsq"]

EPS = 2.0 ** -52
rng = np.random.default_rng(424242)

def ltr(x, c):
    a = x[0] * c[0]
    for i in range(1, len(x)):
        a = a + x[i] * c[i]
    return a

def fr2(v):
    f = Fraction(v); return f * f

pool = dict(K2=[], K1=[], K2tw=[], ratio_bound=[])
tot_rows = 0; reimpl_bad = 0; xx_bad = 0
def run(est, rows, label):
    global tot_rows, reimpl_bad, xx_bad
    C = np.asarray(getattr(est, "cluster_centers_", None) if hasattr(est, "cluster_centers_") else est.subcluster_centers_, dtype=np.float64)
    CC = row_norms(C, squared=True); Cl = [list(map(float, c)) for c in C]
    n = C.shape[1]
    K2s, K1s, K2tws, z1, z2 = [], [], [], 0, 0
    for r in rows:
        x = [float(v) for v in r]
        tw = est.transform([x])[0]
        X = np.array([x])
        re = np.sqrt(np.maximum((-2 * (X @ C.T) + row_norms(X, squared=True)[:, None]) + CC[None, :], 0))[0]
        reimpl_bad += not np.array_equal(re, tw); tot_rows += 1
        xx = row_sumsq(x); xx_bad += xx != float(row_norms(X, squared=True)[0])
        for k in range(len(C)):
            dn = math.sqrt(max((-2.0 * ltr(x, Cl[k]) + xx) + CC[k], 0.0)); dt = float(tw[k])
            S2 = sum(fr2(v) for v in x) + sum(fr2(v) for v in Cl[k]) + 2 * sum(abs(Fraction(a) * Fraction(b)) for a, b in zip(x, Cl[k]))
            qe = sum((Fraction(a) - Fraction(b)) ** 2 for a, b in zip(x, Cl[k]))
            k2 = float(abs(fr2(dn) - fr2(dt)) / S2) / EPS
            K2s.append(k2); K1s.append(abs(dn - dt) / (EPS * math.sqrt(float(S2))))
            K2tws.append(float(abs(fr2(dt) - qe) / S2) / EPS)
            pool["ratio_bound"].append(k2 / (n + 4))
            z1 += (dt == 0.0 and dn != 0.0); z2 += (dn == 0.0 and dt != 0.0)
    for k, v in (("K2", K2s), ("K1", K1s), ("K2tw", K2tws)):
        pool[k] += v
    print(f"{label:<46} lanes={len(K2s):>6} K2max={max(K2s):.3f} K1max={max(K1s):.3g} twin-exact K2max={max(K2tws):.3f} t0/n>0={z1} n0/t>0={z2}", flush=True)

for n in (3, 16, 64):
    for kind in ("lognormal", "unif_offset", "student_t", "integers"):
        if kind == "lognormal":
            X = rng.lognormal(mean=1.0, sigma=1.0, size=(1500, n))
        elif kind == "unif_offset":
            X = rng.uniform(50, 60, size=(1500, n)) + rng.integers(0, 4, size=(1500, 1)) * 3
        elif kind == "student_t":
            X = rng.standard_t(3, size=(1500, n)) + rng.integers(0, 4, size=(1500, 1)) * 5
        else:
            X = rng.poisson(5, size=(1500, n)).astype(float) + rng.integers(0, 3, size=(1500, 1)) * 10
        km = KMeans(n_clusters=6, n_init=1, random_state=1).fit(X)
        C = km.cluster_centers_
        nr = 120 if n < 64 else 60
        rows = np.vstack([
            X[rng.choice(1500, nr, replace=False)],
            C[rng.integers(0, 6, nr)] * (1 + 1e-7 * rng.normal(size=(nr, n))),
            C[rng.integers(0, 6, nr)] + 1e-12 * rng.normal(size=(nr, n)),
            C.copy(),
            np.zeros((1, n)),
            X[:5] * 30,
        ])
        run(km, rows, f"KMeans n={n} {kind}")
Xb = rng.lognormal(0.5, 0.7, size=(1500, 12)) + 40
for est in (MiniBatchKMeans(n_clusters=7, n_init=1, random_state=2), BisectingKMeans(n_clusters=7, random_state=2), Birch(threshold=1.5, n_clusters=None)):
    est.fit(Xb)
    C = np.asarray(getattr(est, "cluster_centers_", None) if hasattr(est, "cluster_centers_") else est.subcluster_centers_)
    rows = np.vstack([Xb[rng.choice(1500, 60, replace=False)], C[rng.integers(0, len(C), 60)] * (1 + 1e-9 * rng.normal(size=(60, 12)))])
    run(est, rows, f"{type(est).__name__} n=12 (k={len(C)})")
a = {k: np.asarray(v) for k, v in pool.items()}
print(f"POOLED lanes={a['K2'].size} rows={tot_rows} reimpl_mismatch_rows={reimpl_bad} xx_mismatch_rows={xx_bad} K2 max={a['K2'].max():.3f} p99.9={np.percentile(a['K2'],99.9):.3f} K1 max={a['K1'].max():.3g} twin-exact K2 max={a['K2tw'].max():.3f} max K2/(n+4)={a['ratio_bound'].max():.3f}")
