"""Accuracy of numpy's float64 log1p/expm1/log/exp (whatever dispatch is
active) and glibc's (ctypes libm), against a correctly rounded reference
(gmpy2/MPFR). Run under different NPY_DISABLE_CPU_FEATURES to compare.

usage: python kernels.py OUT.npz [N]
"""

import ctypes
import os
import sys

import gmpy2
import numpy as np
from numpy._core import _multiarray_umath as m

out = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
libm = ctypes.CDLL("libm.so.6")
for f in ("log1p", "expm1", "log", "exp"):
    getattr(libm, f).restype = ctypes.c_double
    getattr(libm, f).argtypes = [ctypes.c_double]

rng = np.random.default_rng(12345)


def loguni(lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


q = N // 4
inputs = {
    "log1p": np.concatenate([
        loguni(1e-300, 1e300, q),  # positive, full range
        -1 + loguni(1e-16, 1, q),  # (-1, 0), near -1 too
        rng.uniform(0, 1e3, q),  # fixture "plain" |x|
        loguni(1e-20, 1, q) * rng.choice([-1, 1], q) * 0.999,  # small
    ]),
    "expm1": np.concatenate([
        rng.uniform(-40, 709, q),
        loguni(1e-18, 3, q) * rng.choice([-1, 1], q),
        rng.uniform(-3, 3, q) * np.log1p(rng.uniform(0, 1e3, q)),
        rng.uniform(-745, -40, q),
    ]),
    "log": np.concatenate([
        loguni(1e-300, 1e300, 2 * q),
        1 + rng.uniform(-1e-6, 1e-6, q),
        rng.uniform(0.5, 2, q),
    ]),
    "exp": np.concatenate([
        rng.uniform(-745, 709, 2 * q),
        rng.uniform(-1, 1, q),
        loguni(1e-18, 1e-3, q) * rng.choice([-1, 1], q),
    ]),
}

ctx_hp = gmpy2.context(precision=160)
ctx_d = gmpy2.ieee(64)


def ref(f, x):
    """(correctly rounded double, high-precision value)"""
    with gmpy2.context(ctx_hp):
        hp = getattr(gmpy2, f)(gmpy2.mpfr(float(x)))
    with gmpy2.context(ctx_d):
        cr = float(getattr(gmpy2, f)(gmpy2.mpfr(float(x))))
    return cr, hp


def ulp_err(y, hp, cr):
    """|y - exact| in ulps of the correctly rounded result."""
    if not np.isfinite(cr) or cr == 0:
        return 0.0 if y == cr else float("inf")
    sp = np.spacing(abs(cr))
    with gmpy2.context(ctx_hp):
        return float(abs(gmpy2.mpfr(float(y)) - hp) / gmpy2.mpfr(float(sp)))


res = {}
info = m.__cpu_targets_info__
for f, x in inputs.items():
    with np.errstate(all="ignore"):
        npv = getattr(np, f)(x)  # batch, contiguous
        # per element, as the twin calls it (1-element arrays)
        sub = x[:: max(1, len(x) // 20000)]
        per = np.array([getattr(np, f)(np.array([v]))[0] for v in sub])
        bat = getattr(np, f)(sub)
    gl = np.array([getattr(libm, f)(float(v)) for v in x])
    crs = np.empty_like(x)
    e_np = np.empty_like(x)
    e_gl = np.empty_like(x)
    for i, v in enumerate(x):
        cr, hp = ref(f, v)
        crs[i] = cr
        e_np[i] = ulp_err(npv[i], hp, cr)
        e_gl[i] = ulp_err(gl[i], hp, cr)
    fin = np.isfinite(crs) & (crs != 0)
    res[f] = dict(
        target=info[f]["dd"]["current"],
        n=len(x),
        np_ne_glibc=int(np.sum(npv != gl)),
        np_not_cr=int(np.sum(npv[fin] != crs[fin])),
        gl_not_cr=int(np.sum(gl[fin] != crs[fin])),
        np_max_ulp=float(np.max(e_np[fin])),
        gl_max_ulp=float(np.max(e_gl[fin])),
        np_arg=float(x[fin][np.argmax(e_np[fin])]),
        gl_arg=float(x[fin][np.argmax(e_gl[fin])]),
        per_elem_ne_batch=int(np.sum((per != bat) & ~(np.isnan(per) & np.isnan(bat)))),
        per_n=len(sub),
    )
    np.savez(f"{out}_{f}.npz", x=x, npv=npv, gl=gl, cr=crs)
    print(f, res[f], flush=True)

print("NPY_DISABLE_CPU_FEATURES=", os.environ.get("NPY_DISABLE_CPU_FEATURES"))
