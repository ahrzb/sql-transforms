"""Decompose the twin: is scipy boxcox == glibc expm1(lam*log x)/lam?  Then
search lanes where the expm1 values differ by 2 ulps and lambda puts t at the
top of its binade. Run with NPY_DISABLE_CPU_FEATURES=X86_V4."""
import math

import numpy as np
from scipy.special import boxcox

rng = np.random.default_rng(7)


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


x = np.exp(rng.uniform(-5, 5, 1_000_000))
lam = rng.uniform(-3, 3, 1_000_000)
tw = boxcox(x, lam)
dec = np.expm1(lam * np.log(x)) / lam
print("twin == glibc expm1(lam*log x)/lam on", np.mean(tw == dec), "of draws")

best = 0
worst = None
cnt = {}
for lam_sig in [1.0005, 1.001, 1.002, 1.003, 1.01]:
    for m in range(-4, 4):
        for s in (1.0, -1.0):
            lam = s * math.ldexp(lam_sig, m)
            # x whose w gives expm1 with significand in [1, lam_sig)
            p = rng.integers(-40, 0, 400_000)
            sig = 1.0 + rng.uniform(0, lam_sig - 1.0, 400_000)
            e = np.ldexp(sig, p)
            if s < 0:
                e = -e
            w = np.log1p(e)
            x = np.exp(w / lam)
            x = x[np.isfinite(x) & (x > 0)]
            ww = lam * np.log(x)
            e1, e2 = np.expm1(ww), kahan(ww)
            k = ulps(e1, e2)
            t1, t2 = boxcox(x, lam), e2 / lam
            d = ulps(t1, t2)
            for kv, dv in zip(k, d):
                pass
            for kv in (1, 2, 3):
                sel = k == kv
                if sel.any():
                    cnt.setdefault(kv, [0, 0])
                    cnt[kv][0] += int(sel.sum())
                    cnt[kv][1] = max(cnt[kv][1], int(d[sel].max()))
            if d.max() > best:
                i = int(np.argmax(d))
                best = int(d[i])
                worst = dict(x=repr(float(x[i])), lam=repr(lam), w=repr(float(ww[i])),
                             e_glibc=repr(float(e1[i])), e_kahan=repr(float(e2[i])), k=int(k[i]),
                             twin=repr(float(t1[i])), entry=repr(float(t2[i])))
print("expm1 distance k -> (lanes, max ulps of t):", cnt)
print("max ulps of t:", best, worst)
