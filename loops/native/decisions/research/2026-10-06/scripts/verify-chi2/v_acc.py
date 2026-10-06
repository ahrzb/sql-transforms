"""Independent accuracy check of numpy (SVML) log vs glibc log (via DuckDB ln),
reference MPFR at 256 bits."""
import math
import sys

import duckdb
import gmpy2
import numpy as np
import pyarrow as pa

gmpy2.get_context().precision = 256
con = duckdb.connect()


def duck_ln(x):
    t = pa.table({"x": x})  # noqa: F841
    return con.execute("select ln(x) as l from t").to_arrow_table()["l"].to_numpy()


def errs(x, y):
    out = np.empty(len(x))
    for i, (xi, yi) in enumerate(zip(x.tolist(), y.tolist())):
        ex = gmpy2.log(gmpy2.mpfr(xi))
        e = math.frexp(float(ex))[1]  # |ex| in [2^(e-1), 2^e)
        ulp = gmpy2.mpfr(2) ** (e - 53)
        out[i] = float(abs(gmpy2.mpfr(yi) - ex) / ulp)
    return out


def report(name, x):
    a, b = np.log(x), duck_ln(x)
    diff = a != b
    dist = np.abs(a.view(np.int64) - b.view(np.int64))
    # error only where they differ (where equal, errors equal) plus a sample
    idx = np.flatnonzero(diff)
    samp = np.random.default_rng(3).choice(len(x), min(20000, len(x)), replace=False)
    ea, eb = errs(x[idx], a[idx]), errs(x[idx], b[idx])
    sa, sb = errs(x[samp], a[samp]), errs(x[samp], b[samp])
    print(f"{name}: n={len(x)} differ={diff.mean():.3e} ({diff.sum()}) maxdist={dist.max()}"
          f" | on-differ max err np={ea.max() if len(ea) else 0:.4f} glibc={eb.max() if len(eb) else 0:.4f}"
          f" np-worse={int((ea > eb).sum())} | sample max np={sa.max():.4f} glibc={sb.max():.4f}")


rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
report("U(0,1e3)", rng.uniform(0, 1e3, 400_000))
report("U(0.9,1.1)", rng.uniform(0.9, 1.1, 400_000))
report("1+N(0,1e-3)", 1 + rng.normal(0, 1e-3, 400_000))
report("U(0.94,1.07)", rng.uniform(0.94, 1.07, 400_000))
report("1+-k*2^-52 k<2^20", 1 + rng.integers(-2**20, 2**20, 200_000) * 2.0**-52)
m = rng.integers(0, 2**52, 400_000, dtype=np.uint64) | np.uint64(0x3FF0000000000000)
report("randmant full", np.ldexp(m.view(np.float64), rng.integers(-1074, 1023, 400_000)))
