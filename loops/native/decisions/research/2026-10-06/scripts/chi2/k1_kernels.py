"""Q2: numpy float64 log / cos / sin against glibc (math module) and against
a correctly rounded reference (MPFR via gmpy2).

Run: cd /home/user/sql-transforms && uv run --frozen --with gmpy2 python -I \
  <this file>
"""

import json
import math
import sys
import time

import gmpy2
import numpy as np

SEED = 20261006
rng = np.random.default_rng(SEED)
N = 200_000

# A 200-bit context for the "exact" value, and a 53-bit one (with the double
# exponent range and subnormals) for the correctly rounded double.
HI = gmpy2.context(precision=200)
CR = gmpy2.ieee(64)


def cr(fn, x):
    with gmpy2.local_context(CR):
        return float(fn(gmpy2.mpfr(x)))


def err_ulps(fn, xs, ys):
    """(y - exact) / ulp(exact) for each y, exact at 200 bits."""
    out = np.empty(len(xs))
    with gmpy2.local_context(HI):
        for i, (x, y) in enumerate(zip(xs.tolist(), ys.tolist())):
            e = fn(gmpy2.mpfr(x))
            ef = float(e)
            if ef == 0.0:
                out[i] = 0.0 if y == 0.0 else np.inf
                continue
            u = math.ulp(ef)
            out[i] = float((gmpy2.mpfr(y) - e) / u)
    return out


def ordered(a):
    i = a.view(np.int64)
    return np.where(i >= 0, i, -(i & 0x7FFF_FFFF_FFFF_FFFF))


def dist(a, b):
    return np.abs(ordered(a).astype(object) - ordered(b).astype(object)).astype(np.float64)


draws = {
    "uniform(0,1e3)": rng.uniform(0.0, 1e3, N),
    "loguniform(1e-300,1e300)": np.exp(rng.uniform(math.log(1e-300), math.log(1e300), N)),
    "near1 1+N(0,1e-3)": 1.0 + rng.normal(0.0, 1e-3, N),
    "window (0.9,1.1)": rng.uniform(0.9, 1.1, N),
    "subnormal (0,2.2e-308)": rng.uniform(0.0, 2.2250738585072014e-308, N // 4),
}

report = {"numpy": np.__version__, "seed": SEED, "log": {}, "trig": {}}
for name, x in draws.items():
    x = x[x > 0]
    t0 = time.time()
    y_np = np.log(x)
    y_gl = np.array([math.log(v) for v in x.tolist()])
    y_cr = np.array([cr(gmpy2.log, v) for v in x.tolist()])
    e_np = err_ulps(gmpy2.log, x, y_np)
    e_gl = err_ulps(gmpy2.log, x, y_gl)
    d = dist(y_np, y_gl)
    report["log"][name] = {
        "n": int(len(x)),
        "frac numpy != glibc": float(np.mean(y_np != y_gl)),
        "max ulps numpy-glibc": float(d.max()),
        "frac numpy != CR": float(np.mean(y_np != y_cr)),
        "frac glibc != CR": float(np.mean(y_gl != y_cr)),
        "max |err| numpy (ulp)": float(np.abs(e_np).max()),
        "max |err| glibc (ulp)": float(np.abs(e_gl).max()),
        "argmax numpy": float(x[np.argmax(np.abs(e_np))]),
        "secs": round(time.time() - t0, 1),
    }
    print(name, json.dumps(report["log"][name]), flush=True)

# numpy != glibc as a function of |log x| (log-uniform draws)
x = draws["loguniform(1e-300,1e300)"]
L = np.array([math.log(v) for v in x.tolist()])
diff = np.log(x) != L
bins = [0, 0.0625, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
by = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (np.abs(L) >= lo) & (np.abs(L) < hi)
    if m.sum():
        by.append((lo, hi, int(m.sum()), float(diff[m].mean())))
report["log"]["differ_by_absL_loguniform"] = by
print("differ by |L|:", by, flush=True)

# A uniform-in-|L| sweep: x = exp(+-U(0, 709)), to see differ rate at large |L|.
Lt = rng.uniform(0, 709, N) * rng.choice([-1.0, 1.0], N)
x = np.exp(Lt)
x = x[(x > 0) & np.isfinite(x)]
L = np.array([math.log(v) for v in x.tolist()])
diff = np.log(x) != L
by = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (np.abs(L) >= lo) & (np.abs(L) < hi)
    if m.sum():
        by.append((lo, hi, int(m.sum()), float(diff[m].mean())))
report["log"]["differ_by_absL_expuniform"] = by
print("differ by |L| (x=exp(+-U(0,709))):", by, flush=True)

# Row-by-row: one-element arrays, 0-d scalars and strided views equal the
# vectorized call.
x = draws["uniform(0,1e3)"][:20_000]
v = np.log(x)
one = np.array([np.log(x[i : i + 1])[0] for i in range(len(x))])
sc = np.array([np.log(np.float64(t)) for t in x])
xx = np.repeat(x, 3)[::3]
st = np.log(np.repeat(x, 3)[::3])
report["log"]["row_vs_vector_equal"] = bool(
    np.array_equal(v, one) and np.array_equal(v, sc) and np.array_equal(v, st)
)
print("row/scalar/strided == vector:", report["log"]["row_vs_vector_equal"])

# cos / sin: numpy vs glibc (math) on 1e6 arguments: j*s*log(x) for the
# defaults, plus uniform(-1e3,1e3) and huge arguments.
args = [rng.uniform(-1e3, 1e3, 300_000), rng.uniform(-50, 50, 300_000),
        np.exp(rng.uniform(-700, 700, 200_000)) * rng.choice([-1.0, 1.0], 200_000)]
xu = rng.uniform(0, 1e3, 100_000)
xu = xu[xu > 0]
for s, js in ((0.8, (1,)), (0.5, (1, 2)), (0.4, (1, 2, 3))):
    for j in js:
        args.append(j * (s * np.log(xu)))
a = np.concatenate(args)
for fname, f, g in (("cos", np.cos, math.cos), ("sin", np.sin, math.sin)):
    yn = f(a)
    yg = np.array([g(t) for t in a.tolist()])
    report["trig"][fname] = {"n": int(len(a)), "differ": int(np.sum(yn != yg))}
    print(fname, report["trig"][fname], flush=True)

with open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w") as fh:
    json.dump(report, fh, indent=1)
