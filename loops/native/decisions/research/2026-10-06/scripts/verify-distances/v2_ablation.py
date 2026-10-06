"""Is the dot's own term scale 2*sum|x c| enough for the squared lane? Sweep centre scale."""
import math
from fractions import Fraction
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms
src = open("/home/user/sql-transforms/packages/sql-transform/sql_transform/native/_helpers.py").read()
s = src.index("def row_sumsq("); e = src.index("@functools.cache", s)
ns = {}; exec("from typing import Any\n" + src[s:e], ns); row_sumsq = ns["row_sumsq"]
EPS = 2.0 ** -52
rng = np.random.default_rng(5150)
for n in (4, 16, 64):
    for cs in (1e-1, 1e-2, 1e-3):
        C = rng.normal(size=(8, n)) * cs
        Cl = [list(map(float, c)) for c in C]; CC = row_norms(C, squared=True)
        Kd, K2, diff = [], [], 0
        for _ in range(1500):
            x = list(map(float, rng.normal(size=n) * 10))
            tw = euclidean_distances(np.array([x]), C)[0]; xx = row_sumsq(x)
            for k in range(8):
                a = x[0] * Cl[k][0]
                for i in range(1, n): a = a + x[i] * Cl[k][i]
                d = math.sqrt(max((-2.0 * a + xx) + CC[k], 0.0)); t = float(tw[k])
                if d == t: continue
                diff += 1
                dq = abs(float(Fraction(d) ** 2 - Fraction(t) ** 2))
                sd = 2 * math.fsum(abs(p * q) for p, q in zip(x, Cl[k]))
                Kd.append(dq / (EPS * sd)); K2.append(dq / (EPS * (sd + math.fsum(v * v for v in x) + CC[k])))
        print(f"n={n:>2} centre scale {cs:g}: lanes 12000, differ {diff}; max K with 2sum|xc| = {max(Kd) if Kd else 0:.3g}; median over differing = {np.median(Kd) if Kd else 0:.3g}; max K2 with S2 = {max(K2) if K2 else 0:.3f}")
