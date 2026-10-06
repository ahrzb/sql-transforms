"""Is the SERVED Box-Cox bound (power.py: ulps=4, a measurement) breachable?

Twin: scipy.special.boxcox (glibc log/expm1), exactly what sklearn calls.
Entry: power.py's spelling, with glibc exp/ln (numpy with X86_V4 disabled is
glibc bit for bit; verified on a sample below against math.exp/math.log).

Targeted: lambda = 2^m * (1+delta) so t = e/lambda lands near the top of its
binade while e = expm1(w) sits near the bottom of its own: a k-ulp distance in
e becomes ~2k ulps of t, plus one rounding per side.
Run with NPY_DISABLE_CPU_FEATURES=X86_V4.
"""
import math
import sys

import numpy as np
from scipy.special import boxcox

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)


def ordered(a):
    i = a.view(np.int64)
    return np.where(i >= 0, i, -(i & 0x7FFF_FFFF_FFFF_FFFF))


def ulps(a, b):
    return np.abs(ordered(a) - ordered(b))


def entry(x, lam):
    lx = np.log(x)
    w = lam * lx
    u = np.exp(w)
    with np.errstate(all="ignore"):
        k = (u - 1.0) * (w / np.log(u))
    k = np.where(u == 1.0, w, k)
    k = np.where(u - 1.0 == -1.0, -1.0, k)
    e = k / lam
    big = np.exp(w - math.log(abs(lam)))
    big = np.where(lam > 0, big, -big) - 1.0 / lam
    return np.where(w < 709.78, e, big)


# sanity: numpy exp/log are glibc here
s = rng.uniform(-700, 700, 20000)
assert all(math.exp(v) == np.exp(np.array([v]))[0] for v in s[:5000]), "np.exp != glibc"
p = rng.uniform(1e-3, 1e3, 20000)
assert all(math.log(v) == np.log(np.array([v]))[0] for v in p[:5000]), "np.log != glibc"

best = (0, None)
hist = {}
total = 0
for trial in range(400):
    m = rng.integers(-4, 4)
    delta = rng.choice([1e-9, 1e-7, 1e-5, 1e-3, 3e-3])
    lam = float(np.ldexp(1.0 + delta, int(m))) * rng.choice([1.0, -1.0])
    # target e near the bottom of a binade: e = 2^p (1 + small)
    pexp = rng.integers(-30, 40)
    e_target = np.ldexp(1.0 + rng.uniform(0, 2e-3, 50000), int(pexp))
    if lam < 0:
        e_target = np.ldexp(1.0 + rng.uniform(0, 2e-3, 50000), int(min(pexp, -1)))
        e_target = -e_target  # expm1 < 0 when w < 0; magnitude < 1
        e_target = np.clip(e_target, -0.999999, -1e-300)
    w_target = np.log1p(e_target)
    x = np.exp(w_target / lam)
    ok = np.isfinite(x) & (x > 0)
    x = x[ok]
    if x.size == 0:
        continue
    t_twin = boxcox(x, lam)
    t_ent = entry(x, lam)
    fin = np.isfinite(t_twin) & np.isfinite(t_ent)
    d = ulps(t_twin[fin], t_ent[fin])
    total += int(fin.sum())
    for v, c in zip(*np.unique(d, return_counts=True)):
        hist[int(v)] = hist.get(int(v), 0) + int(c)
    i = int(np.argmax(d)) if d.size else 0
    if d.size and d[i] > best[0]:
        xi = x[fin][i]
        best = (int(d[i]), (repr(float(xi)), repr(lam), repr(float(t_twin[fin][i])), repr(float(t_ent[fin][i]))))

print("lanes", total)
print("ulps histogram", dict(sorted(hist.items())))
print("max ulps", best[0], "at (x, lambda, twin, entry) =", best[1])
