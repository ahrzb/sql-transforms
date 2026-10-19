"""(a) Is the dot product's own term scale 2*sum|x_i c_i| enough for the squared
lane, or does S need ||x||^2 + ||c||^2 too?  Centres near the origin, rows far.
(b) KMeans fitted on float32: does the twin stay on the float64 path?"""
import math, sys
from fractions import Fraction
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rowsumsq import row_sumsq
EPS = 2.0 ** -52
rng = np.random.default_rng(23)
for n in (4, 16, 64):
    C = rng.normal(size=(8, n)) * 1e-3          # centres near the origin
    Cl = [list(map(float, c)) for c in C]; CC = row_norms(C, squared=True)
    Kdot, K2, diff = [], [], 0
    for _ in range(2000):
        x = list(map(float, rng.normal(size=n) * 10))
        twin = euclidean_distances(np.asarray([x]), C)[0]
        xx = row_sumsq(x)
        for k in range(8):
            acc = x[0] * Cl[k][0]
            for i in range(1, n):
                acc = acc + x[i] * Cl[k][i]
            d = math.sqrt(max((-2.0 * acc + xx) + CC[k], 0.0))
            dq = abs(float(Fraction(d) ** 2 - Fraction(float(twin[k])) ** 2))
            sdot = 2 * math.fsum(abs(a * b) for a, b in zip(x, Cl[k]))
            s2 = sdot + math.fsum(v * v for v in x) + CC[k]
            Kdot.append(dq / (EPS * sdot)); K2.append(dq / (EPS * s2)); diff += d != float(twin[k])
    print(f"(a) n={n:>2}: lanes {len(K2)}, differ {diff}; K with S=2sum|xc| max {max(Kdot):.3g}; K2 with S2 max {max(K2):.3f}")
# (b)
X = rng.normal(size=(500, 6)).astype(np.float32)
km = KMeans(n_clusters=4, n_init=1, random_state=0).fit(X)
x = [float(v) for v in rng.normal(size=6)]
t = km.transform([x])
ref = euclidean_distances(np.asarray([x]), km.cluster_centers_.astype(np.float64))
print(f"(b) centres dtype {km.cluster_centers_.dtype}; twin output dtype {t.dtype}; equals float64-upcast path: {np.array_equal(t, ref)}")
