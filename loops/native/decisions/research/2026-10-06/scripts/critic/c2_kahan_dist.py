"""Where does Kahan's expm1 (on glibc exp/ln) sit 2+ ulps from glibc expm1?
Then: amplify through t = e / lambda with lambda's significand just above 1.
Run with NPY_DISABLE_CPU_FEATURES=X86_V4 (numpy exp/log/expm1 = glibc)."""
import math

import numpy as np
from scipy.special import boxcox

rng = np.random.default_rng(1)


def ordered(a):
    i = a.view(np.int64)
    return np.where(i >= 0, i, -(i & 0x7FFF_FFFF_FFFF_FFFF))


def ulps(a, b):
    return np.abs(ordered(a) - ordered(b))


def kahan(w):
    u = np.exp(w)
    with np.errstate(all="ignore"):
        k = (u - 1.0) * (w / np.log(u))
    k = np.where(u == 1.0, w, k)
    return np.where(u - 1.0 == -1.0, -1.0, k)


assert math.expm1(0.3) == np.expm1(np.array([0.3]))[0]
print("w-bin            n       frac>=2   max  ")
for lo, hi in [(1e-18, 1e-12), (1e-12, 1e-6), (1e-6, 1e-3), (1e-3, 0.1), (0.1, 1), (1, 3), (3, 40), (40, 700)]:
    for sgn in (1, -1):
        w = sgn * np.exp(rng.uniform(math.log(lo), math.log(hi), 2_000_000))
        d = ulps(kahan(w), np.expm1(w))
        print(f"{sgn:+d} [{lo:g},{hi:g})  {w.size}  {np.mean(d >= 2):.2e}  {d.max()}")

# amplification through /lambda: take w where d>=2, choose lambda = 2^m(1+eps')
# with e/lambda near the top of its binade, solve x = exp(w/lambda).
w = np.exp(rng.uniform(math.log(1e-12), math.log(1.0), 20_000_000))
w = np.concatenate([w, -w])
d = ulps(kahan(w), np.expm1(w))
w2 = w[d >= 2]
print("w with Kahan-glibc >= 2 ulps:", w2.size)
best = 0
worst = None
for lam_sig in [1 + 1e-12, 1 + 1e-9, 1 + 1e-6, 1 + 1e-4, 1.001, 1.01]:
    for m in range(-3, 4):
        for s in (1.0, -1.0):
            lam = s * math.ldexp(lam_sig, m)
            x = np.exp(w2 / lam)
            ok = np.isfinite(x) & (x > 0)
            x = x[ok]
            t1 = boxcox(x, lam)
            # entry: w recomputed from x as the entry does
            lx = np.log(x)
            ww = lam * lx
            t2 = kahan(ww) / lam
            dd = ulps(t1, t2)
            if dd.size and dd.max() > best:
                i = int(np.argmax(dd))
                best = int(dd[i])
                worst = (repr(float(x[i])), repr(lam), repr(float(t1[i])), repr(float(t2[i])))
print("max ulps of t", best, worst)
