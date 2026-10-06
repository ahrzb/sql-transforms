"""Goldberg's log1p and Kahan's expm1, spelled from glibc's log/exp as a
DuckDB entry would, against a correctly rounded reference (MPFR via gmpy2),
glibc's log1p/expm1, and numpy's (SVML HA on this CPU). The same spellings
are run on DuckDB over the same draws and compared bit for bit.

usage: python spellings.py N
"""

import ctypes
import sys

import duckdb
import gmpy2
import numpy as np
import pyarrow as pa

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
libm = ctypes.CDLL("libm.so.6")
for f in ("log1p", "expm1", "log", "exp"):
    getattr(libm, f).restype = ctypes.c_double
    getattr(libm, f).argtypes = [ctypes.c_double]
ln, ex = libm.log, libm.exp


def goldberg_log1p(z):
    u = 1.0 + z
    if u == 1.0:
        return z
    return z * (ln(u) / (u - 1.0))


def kahan_expm1(w):
    if w > 709.782712893384:  # exp overflows; the twin's expm1 is inf too
        return float("inf")
    u = ex(w)
    if u == 1.0:
        return w
    if u - 1.0 == -1.0:
        return -1.0
    return (u - 1.0) * (w / ln(u))


rng = np.random.default_rng(7)


def loguni(lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


q = N // 4
Z = np.concatenate([  # log1p arguments in Yeo-Johnson are >= 0
    loguni(1e-300, 1e300, q), rng.uniform(0, 1e3, q), loguni(1e-20, 1, q),
    rng.uniform(0, 0.75, q),
])
W = np.concatenate([
    rng.uniform(-40, 709, q), loguni(1e-18, 3, q) * rng.choice([-1, 1], q),
    rng.uniform(-3, 3, q) * np.log1p(rng.uniform(0, 1e3, q)),
    rng.uniform(-745, -40, q),
])
hp = gmpy2.context(precision=160)
dbl = gmpy2.ieee(64)


def stats(name, xs, mine, fname, npf):
    glib = np.array([getattr(libm, fname)(float(v)) for v in xs])
    npv = npf(xs)
    errs = np.empty(len(xs))
    errs_np = np.empty(len(xs))
    crs = np.empty(len(xs))
    for i, v in enumerate(xs):
        with gmpy2.context(hp):
            e = getattr(gmpy2, fname)(gmpy2.mpfr(float(v)))
        with gmpy2.context(dbl):
            cr = float(getattr(gmpy2, fname)(gmpy2.mpfr(float(v))))
        crs[i] = cr
        if not np.isfinite(cr) or cr == 0:
            errs[i] = errs_np[i] = 0.0 if mine[i] == cr else np.inf
            continue
        sp = gmpy2.mpfr(float(np.spacing(abs(cr))))
        with gmpy2.context(hp):
            errs[i] = float(abs(gmpy2.mpfr(float(mine[i])) - e) / sp)
            errs_np[i] = float(abs(gmpy2.mpfr(float(npv[i])) - e) / sp)

    def ud(a, b):
        ia = a.view(np.int64).astype(object)
        ib = b.view(np.int64).astype(object)
        f = lambda i: i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)  # noqa: E731
        return np.array([abs(f(x) - f(y)) for x, y in zip(ia, ib)])

    fin = np.isfinite(glib) & np.isfinite(mine)
    d_gl = ud(mine[fin], glib[fin])
    d_np = ud(mine[fin], npv[fin])
    print(f"{name}: n={len(xs)}  err vs exact: max {errs.max():.3f} ulp,"
          f" p99.9 {np.quantile(errs, 0.999):.3f}, not CR {np.mean(mine != crs):.4%}"
          f" | dist to glibc {fname}: max {d_gl.max()} ulp, differ {np.mean(d_gl > 0):.3%}"
          f" | dist to numpy {fname}: max {d_np.max()} ulp, differ {np.mean(d_np > 0):.3%}"
          f" | numpy err max {errs_np.max():.3f}"
          f" | worst arg {xs[np.argmax(errs)]!r}")
    return mine


with np.errstate(all="ignore"):
    g = np.array([goldberg_log1p(float(z)) for z in Z])
    k = np.array([kahan_expm1(float(w)) for w in W])
stats("goldberg log1p", Z, g, "log1p", np.log1p)
stats("kahan expm1", W, k, "expm1", np.expm1)

# The same spellings on DuckDB, bit for bit.
con = duckdb.connect()
con.register("z", pa.table({"z": Z}))
con.register("w", pa.table({"w": W}))
gd = con.sql(
    "SELECT CASE WHEN 1.0::DOUBLE + z = 1.0 THEN z"
    " ELSE z * (ln(1.0::DOUBLE + z) / ((1.0::DOUBLE + z) - 1.0)) END FROM z"
).fetchnumpy()
gd = list(gd.values())[0]
kd = con.sql(
    "SELECT CASE WHEN w > 709.782712893384 THEN 'inf'::DOUBLE"
    " WHEN exp(w) = 1.0 THEN w WHEN exp(w) - 1.0 = -1.0 THEN -1.0"
    " ELSE (exp(w) - 1.0) * (w / ln(exp(w))) END FROM w"
).fetchnumpy()
kd = list(kd.values())[0]
print("duckdb goldberg == python:", int(np.sum(gd.view(np.int64) != g.view(np.int64))), "mismatches of", len(g))
print("duckdb kahan == python:", int(np.sum(kd.view(np.int64) != k.view(np.int64))), "mismatches of", len(k))
