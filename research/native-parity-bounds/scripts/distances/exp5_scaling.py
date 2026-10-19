"""K versus the number of features n (the length of the dot product).
Distances: twin = KMeans-fitted centres, euclidean_distances on a 1-row X (what
_BaseKMeans._transform runs after validation); native-like as in exp1.
RBF: twin = RBFSampler.transform on a 1-row list; native-like as in exp3.
Rows: half far (data), half near a centre (1e-6 relative)."""
import math, sys, time
from fractions import Fraction
import numpy as np
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rowsumsq import row_sumsq

EPS = 2.0 ** -52
rng = np.random.default_rng(17)
t0 = time.time()
print(f"{'n':>4} | dist lanes  K2 max  K2 p99.9  K2 p99 | rbf lanes  K max  K p99.9")
for n in (2, 8, 32, 128, 512):
    X = rng.normal(size=(1500, n)) * 3 + rng.normal(size=(1, n)) * 20
    km = KMeans(n_clusters=8, n_init=1, random_state=0, max_iter=50).fit(X)
    C = km.cluster_centers_; Cl = [list(map(float, c)) for c in C]; CC = row_norms(C, squared=True)
    nrows = max(200, 60000 // (n * 8))
    nrows = min(nrows, 1500)
    rows = np.vstack([X[rng.choice(1500, nrows // 2, replace=False)],
                      C[rng.integers(0, 8, nrows // 2)] * (1 + 1e-6 * rng.normal(size=(nrows // 2, n)))])
    K2 = []
    for r in rows:
        x = [float(v) for v in r]
        twin = euclidean_distances(np.asarray([x]), C)[0]
        xx = row_sumsq(x)
        for k in range(8):
            acc = x[0] * Cl[k][0]
            for i in range(1, n):
                acc = acc + x[i] * Cl[k][i]
            d = math.sqrt(max((-2.0 * acc + xx) + CC[k], 0.0))
            s2 = math.fsum(v * v for v in x) + CC[k] + 2 * math.fsum(abs(a * b) for a, b in zip(x, Cl[k]))
            K2.append(abs(float(Fraction(d) ** 2 - Fraction(float(twin[k])) ** 2)) / (EPS * s2))
    K2 = np.array(K2)
    # RBF
    Xs = rng.normal(size=(1000, n))
    est = RBFSampler(gamma=1.0 / n, n_components=100, random_state=n).fit(Xs)
    W = est.random_weights_; b = est.random_offset_; c = (2.0 / 100) ** 0.5
    KR = []
    for r in Xs[: max(100, min(1000, 40000 // n))]:
        x = [float(v) for v in r]
        twin = est.transform([x])[0]
        acc = x[0] * W[0]
        for i in range(1, n):
            acc = acc + x[i] * W[i]
        nat = np.array([math.cos(t) for t in acc + b]) * c
        S = c * (np.abs(np.asarray(x)[:, None] * W).sum(axis=0) + np.abs(b) + 1)
        KR.append(np.abs(nat - twin) / (EPS * S))
    KR = np.concatenate(KR)
    print(f"{n:>4} | {K2.size:>10} {K2.max():7.3f} {np.percentile(K2, 99.9):8.3f} {np.percentile(K2, 99):6.3f} | "
          f"{KR.size:>9} {KR.max():6.3f} {np.percentile(KR, 99.9):7.3f}", flush=True)
print(f"elapsed {time.time()-t0:.1f}s")
