"""Q3, kernel-independent: K when the twin's log is exactly one ulp from
glibc's (L' = nextafter(L, +-inf)) on EVERY draw -- the worst case for any
log kernel within 1 ulp of glibc's (SVML's measured max distance is 1).

Also the argument amplification A = |a' - a| / (eps * j * s * |L|),
a = fl(j * fl(s * L)), whose supremum the derivation predicts:
  s = 0.5 (j = 1, 2): A <= 1 ; s = 0.4 or 0.8: A < 1.25 ; s just above a
  power of two: A -> 2 ; j = 3 adds another rounding.

cos and sin are numpy's (glibc's: measured identical, k1_kernels.py).
Run: cd /home/user/sql-transforms && uv run --frozen python -I <this> <out.json>
"""

import json
import math
import sys

import duckdb
import numpy as np
import pyarrow as pa

EPS = 2.0**-52
rng = np.random.default_rng(11)


def glibc_log(x):
    t = pa.table({"x": x})
    y = duckdb.sql("SELECT ln(x) AS y FROM t").arrow()
    y = y.read_all() if hasattr(y, "read_all") else y
    return y.column("y").to_numpy()


def draws(n):
    # random mantissa, exponent so that |log x| spans 2^-6 .. 745, both signs
    k = rng.integers(-1074 + 53, 1024, n)
    x = np.ldexp(rng.uniform(1.0, 2.0, n), k)
    # plus a dense near-1 block (small X) and |L| just above powers of two
    m = n // 4
    x1 = 1.0 + rng.uniform(-0.5, 0.5, m) * 10.0 ** rng.uniform(-6, 0, m)
    e = rng.integers(-3, 10, m)
    Lp = 2.0 ** e * (1 + rng.uniform(0, 0.02, m)) * rng.choice([-1.0, 1.0], m)
    Lp = Lp[np.abs(Lp) < 700]
    x2 = np.exp(Lp) * (1.0 + rng.uniform(-(2.0**-30), 2.0**-30, len(Lp)))
    x = np.concatenate([x, x1, x2])
    return x[(x > 0) & np.isfinite(x)]


def run(s, j, x, L):
    c = float(np.cosh(np.pi * j * s))
    f = np.sqrt(((2.0 * x) * s) / c)
    a = j * (s * L)
    X = np.abs(a)
    out = {"Amax": 0.0}
    Kmax = {}
    for direction in (np.inf, -np.inf):
        Lq = np.nextafter(L, direction)
        aq = j * (s * Lq)
        A = np.abs(aq - a) / (EPS * j * s * np.abs(L))
        out["Amax"] = max(out["Amax"], float(np.nanmax(A)))
        for lane, fn in (("cos", np.cos), ("sin", np.sin)):
            y, yq = f * fn(a), f * fn(aq)
            K = np.abs(yq - y) / (EPS * f * (1.0 + X))
            ok = np.isfinite(K) & (f > 0)
            for lo, hi in ((0, 0.1), (0.1, 1), (1, 10), (10, 100), (100, 1e4)):
                m = ok & (X >= lo) & (X < hi)
                if m.any():
                    key = f"{lane} X in [{lo},{hi})"
                    Kmax[key] = max(Kmax.get(key, 0.0), float(K[m].max()))
    out["Kmax_by_X"] = Kmax
    out["Kmax"] = max(Kmax.values())
    return out


def main(path):
    x = draws(1_000_000)
    L = glibc_log(x)
    print("draws", len(x), flush=True)
    rep = {"n": int(len(x))}
    cfgs = [(0.5, 1), (0.4, 1), (0.4, 2), (0.8, 1), (0.5, 2), (0.4, 3),
            (0.26, 1), (0.26, 3), (0.5000001, 1), (0.2500001, 2), (1.0000001, 1),
            (0.7, 3), (0.3, 5)]
    for s, j in cfgs:
        r = run(s, j, x, L)
        rep[f"s={s} j={j}"] = r
        print(f"s={s} j={j} Amax={r['Amax']:.4f} Kmax={r['Kmax']:.4f}",
              {k: round(v, 4) for k, v in r["Kmax_by_X"].items()}, flush=True)
    # a sweep of s: sup A over random s in (0.05, 2), j = 1, 2
    sw = []
    for s in rng.uniform(0.05, 2.0, 60):
        for j in (1, 2):
            sw.append(run(float(s), j, x[:300_000], L[:300_000])["Amax"])
    rep["sweep_Amax_max"] = max(sw)
    print("sweep: max A over 60 random s in (0.05,2), j=1,2:", max(sw))
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
