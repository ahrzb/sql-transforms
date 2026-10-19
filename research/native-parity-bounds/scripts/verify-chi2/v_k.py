"""Independent check of K for AdditiveChi2Sampler: twin = sklearn, native =
DuckDB SQL (glibc ln/cos/sin), all constants passed as DOUBLE columns."""
import json
import sys

import duckdb
import numpy as np
import pyarrow as pa
from sklearn.kernel_approximation import AdditiveChi2Sampler

EPS = 2.0**-52
con = duckdb.connect()


def rand_mantissa(rng, n, lo_exp, hi_exp):
    m = rng.integers(0, 2**52, n, dtype=np.uint64) | np.uint64(0x3FF0000000000000)
    return np.ldexp(m.view(np.float64), rng.integers(lo_exp, hi_exp + 1, n))


def lnx_in(rng, n, lo, hi):
    """x with ln x roughly uniform in [lo, hi) (either sign handled by caller),
    then randomized in the last 30 bits of the mantissa to break exp(t) bias."""
    t = rng.uniform(lo, hi, n)
    x = np.exp(t)
    bits = x.view(np.uint64) ^ rng.integers(0, 2**30, n, dtype=np.uint64)
    return bits.view(np.float64)


def duck_ln(x):
    t = pa.table({"x": x})
    return con.execute("select ln(x) as l from t").to_arrow_table()["l"].to_numpy()


def native(x, steps, s):
    cols = {"x": x}
    sel = ["sqrt(x * s) as l0"]
    for j in range(1, steps):
        cols[f"c{j}"] = np.full(len(x), float(np.cosh(np.pi * j * s)))
        cols[f"j{j}"] = np.full(len(x), float(j))
        sel.append(f"sqrt(((2.0 * x) * s) / c{j}) * cos(j{j} * (s * ln(x))) as cos{j}")
        sel.append(f"sqrt(((2.0 * x) * s) / c{j}) * sin(j{j} * (s * ln(x))) as sin{j}")
    cols["s"] = np.full(len(x), s)
    t = pa.table(cols)  # noqa: F841
    out = con.execute("select " + ", ".join(sel) + " from t").to_arrow_table()
    return np.column_stack([out[c].to_numpy() for c in out.column_names])


def twin(x, steps, s):
    return AdditiveChi2Sampler(sample_steps=steps, sample_interval=s).fit_transform(x[:, None])


def kstats(x, steps, s):
    tw, nt = twin(x, steps, s), native(x, steps, s)
    res = {"lane0_diff": int((tw[:, 0] != nt[:, 0]).sum())}
    L = np.log(x)
    for j in range(1, steps):
        f = np.sqrt((2 * x * s) / np.cosh(np.pi * j * s))
        a = j * (s * L)
        S = EPS * f * (1 + np.abs(a))
        for k, name in ((2 * j - 1, "cos"), (2 * j, "sin")):
            d = np.abs(nt[:, k] - tw[:, k])
            K = d / S
            res[f"{name}{j}"] = {
                "lanes_diff": int((d > 0).sum()),
                "maxK": float(K.max()),
                "nK>1": int((K > 1).sum()),
                "x_at_max": float(x[K.argmax()]),
            }
    return res


def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 99)
    sets = {}
    sets["U(0,1e3)"] = rng.uniform(0, 1e3, 1_000_000)
    sets["randmant_full"] = rand_mantissa(rng, 1_000_000, -1021, 1020)
    # mismatch-enriched: candidates over many |ln x| scales, keep np.log != glibc
    cand = np.concatenate([
        rand_mantissa(rng, 6_000_000, -1021, 1020),
        lnx_in(rng, 3_000_000, 2, 709) ,
        1 / lnx_in(rng, 3_000_000, 2, 708),
        lnx_in(rng, 2_000_000, 0, 2),
    ])
    cand = cand[(cand > 0) & np.isfinite(cand)]
    keep = np.log(cand) != duck_ln(cand)
    if keep.any():
        sets["ENR"] = cand[keep]
    out = {"n_enr": int(keep.sum()), "n_cand": int(len(cand))}
    for steps, s in [(2, 0.5), (3, 0.4), (2, 0.8), (3, 0.5), (4, 0.4), (4, 0.26), (2, 0.3), (2, 0.7)]:
        for name, x in sets.items():
            out[f"steps={steps},s={s},{name}"] = kstats(x, steps, s)
    print(json.dumps(out, indent=1))


main()
