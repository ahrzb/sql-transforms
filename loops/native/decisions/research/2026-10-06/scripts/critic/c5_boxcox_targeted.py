"""Bulk search, served Box-Cox (power.py, bound 4 ulps): random lambda, x
chosen so |w| in [1e-12, 1] (where Kahan-vs-glibc reaches 2 ulps), twin =
scipy.special.boxcox, entry = power.py spelling on glibc exp/ln.
Run with NPY_DISABLE_CPU_FEATURES=X86_V4."""
import math
import sys

import numpy as np
from scipy.special import boxcox

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 20
rng = np.random.default_rng(seed)


def ordered(a):
    i = a.view(np.int64)
    return np.where(i >= 0, i, -(i & 0x7FFF_FFFF_FFFF_FFFF))


def ulps(a, b):
    return np.abs(ordered(a) - ordered(b))


def entry(x, lam):
    w = lam * np.log(x)
    u = np.exp(w)
    with np.errstate(all="ignore"):
        k = (u - 1.0) * (w / np.log(u))
    k = np.where(u == 1.0, w, k)
    k = np.where(u - 1.0 == -1.0, -1.0, k)
    return k / lam


hist = {}
best = (0, None)
N = 10_000_000
for c in range(chunks):
    lam = np.ldexp(rng.uniform(1, 1.06, N), rng.integers(-6, 3, N)) * rng.choice([-1.0, 1.0], N)
    e = np.ldexp(rng.uniform(1, 1.04, N), rng.integers(-40, 0, N)) * rng.choice([-1.0, 1.0], N); w = np.log1p(np.maximum(e, -0.999))
    x = np.exp(w / lam)
    ok = np.isfinite(x) & (x > 0) & (x != 1.0)
    x, lam = x[ok], lam[ok]
    t1 = boxcox(x, lam)
    t2 = entry(x, lam)
    d = ulps(t1, t2)
    for v, n in zip(*np.unique(d, return_counts=True)):
        hist[int(v)] = hist.get(int(v), 0) + int(n)
    i = int(np.argmax(d))
    if d[i] > best[0]:
        best = (int(d[i]), dict(x=repr(float(x[i])), lam=repr(float(lam[i])),
                                twin=repr(float(t1[i])), entry=repr(float(t2[i]))))
print("seed", seed, "lanes", sum(hist.values()))
print("ulps histogram", dict(sorted(hist.items())))
print("max", best)
