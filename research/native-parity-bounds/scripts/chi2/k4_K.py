"""Q3: the native lane (confit, served from the SQL an entry would spell) vs
the twin (sklearn 1.9 AdditiveChi2Sampler._transform_dense), ulps and
K = |native - twin| / (eps * factor * (1 + |j s log x|)).

Draw sets (all x with random low mantissa bits, see the note on exp(U)):
  R1 uniform(0, 1e3)                                  (the record's row 1)
  R2 x = exp(U(ln 1e-300, ln 1e300))                  (the record's row 2, biased)
  R2b x = m * 2^k, m ~ U[1,2), k ~ U{-996..996}        (row 2 without the bias)
  R3 1 + N(0, 1e-3)                                   (the record's row 3)
  ADV near zeros of each lane, |log x| just above a power of two
  ENR mismatch-enriched: candidates near zeros kept only where numpy's log
      != glibc's (the real kernels, conditioned on a mismatch)

Run: cd /home/user/sql-transforms && uv run --frozen python -I <this> <out.json>
"""

import json
import os
import math
import sys
import time

import duckdb
import numpy as np
import pyarrow as pa
from confit import DuckDBInferFn
from confit import sql as S
from sklearn.kernel_approximation import AdditiveChi2Sampler

EPS = 2.0**-52
rng = np.random.default_rng(20261006)

# (sample_steps, sample_interval) -> s, js. Defaults: steps 2 -> s 0.5 (j=1),
# steps 3 -> s 0.4 (j=1,2). Plus the record's combos and two explicit ones.
CONFIGS = [
    (2, None),  # default: s=0.5, j=1
    (3, None),  # s=0.4, j=1,2
    (2, 0.8),   # the record's "s=0.8, j=1"
    (3, 0.5),   # the record's T11 "s=0.5, j=1,2"
    (4, 0.4),   # the record's "s=0.4, j=1,2,3"
    (4, 0.26),  # an explicit s just above a power of two, j up to 3
]
DEFAULT_S = {1: 0.8, 2: 0.5, 3: 0.4}


def lanes_sql(steps, s):
    x = S.col("x")
    L = S.fn("ln", x)
    out = {"sqrt": S.fn("sqrt", x * S.lit(s))}
    for j in range(1, steps):
        c = float(np.cosh(np.pi * j * s))
        f = S.fn("sqrt", ((S.lit(2.0) * x) * S.lit(s)) / S.lit(c))
        a = S.lit(float(j)) * (S.lit(s) * L)
        out[f"cos{j}"] = f * S.fn("cos", a)
        out[f"sin{j}"] = f * S.fn("sin", a)
    return out


def native(x, steps, s):
    exprs = lanes_sql(steps, s)
    sel = ", ".join(f"{e.sql()} AS {k}" for k, e in exprs.items())
    sql = f"SELECT {sel} FROM __THIS__"
    rows = pa.table({"x": x})
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[])
    got = fn.infer_arrow(rows)
    duck = duckdb.sql(sql.replace("__THIS__", "rows")).arrow()
    if hasattr(duck, "read_all"):
        duck = duck.read_all()
    res = {}
    for k in exprs:
        a = got.column(k).to_numpy()
        b = duck.column(k).to_numpy()
        assert np.array_equal(a, b), f"confit != duckdb on {k}"
        res[k] = a
    return res


def twin(x, steps, s):
    est = AdditiveChi2Sampler(sample_steps=steps, sample_interval=s).fit(x[:1, None])
    y = est.transform(x[:, None])
    n = len(x)
    out = {"sqrt": y[:, 0]}
    for j in range(1, steps):
        out[f"cos{j}"] = y[:, (2 * j - 1)]
        out[f"sin{j}"] = y[:, (2 * j)]
    return out


def ordered(a):
    i = a.view(np.int64)
    return np.where(i >= 0, i, -(i & 0x7FFF_FFFF_FFFF_FFFF))


def ulps(a, b):
    """Exact distance in doubles (Python ints where int64 could overflow),
    returned as float64 (exact below 2**53)."""
    oa, ob = ordered(a), ordered(b)
    same = (oa >= 0) == (ob >= 0)
    d = np.zeros(len(a), dtype=np.float64)
    d[same] = np.abs(oa[same] - ob[same]).astype(np.float64)
    for i in np.flatnonzero(~same):
        d[i] = float(abs(int(oa[i]) - int(ob[i])))
    return d


def randomize(x):
    """Random low mantissa bits: x * (1 + U(-2^-30, 2^-30)), rounded."""
    return x * (1.0 + rng.uniform(-(2.0**-30), 2.0**-30, len(x)))


def adversarial(s, j, n, lane):
    """x whose j*s*log(x) is within 10^U(-9,-1) of a zero of the lane, with
    |log x| in [2^e, 1.25*2^e) where possible, e = -1..9, both signs."""
    xs = []
    for e in range(-1, 10):
        lo, hi = 2.0**e, 1.25 * 2.0**e
        if hi > 700:
            hi = 700.0
        # zeros of cos: (k+1/2)pi ; of sin: k pi (k != 0)
        a_lo, a_hi = j * s * lo, j * s * hi
        if lane == "cos":
            ks = np.arange(math.floor(a_lo / math.pi - 0.5), math.ceil(a_hi / math.pi - 0.5) + 1)
            zeros = (ks + 0.5) * math.pi
        else:
            ks = np.arange(math.floor(a_lo / math.pi), math.ceil(a_hi / math.pi) + 1)
            zeros = ks[ks != 0] * math.pi
        zeros = zeros[(zeros >= a_lo * 0.9) & (zeros <= a_hi * 1.1)]
        if len(zeros) == 0:
            continue
        m = n // 11
        z = rng.choice(zeros, m)
        d = rng.choice([-1.0, 1.0], m) * 10.0 ** rng.uniform(-9, -1, m)
        L = (z + d) / (j * s)
        sign = rng.choice([-1.0, 1.0], m)
        xs.append(randomize(np.exp(sign * L)))
    x = np.concatenate(xs)
    return x[(x > 0) & np.isfinite(x)]


_CHECKED = []


def glibc_log(x):
    """glibc's log through DuckDB's ln (checked equal to math.log once)."""
    t = pa.table({"x": x})
    y = duckdb.sql("SELECT ln(x) AS y FROM t").arrow()
    if hasattr(y, "read_all"):
        y = y.read_all()
    y = y.column("y").to_numpy()
    if not _CHECKED:
        k = min(len(x), 200_000)
        assert np.array_equal(y[:k], np.array([math.log(v) for v in x[:k].tolist()]))
        _CHECKED.append(True)
    return y


def enriched(s, j, lane, want, cap):
    """Candidates near zeros (as adversarial), kept where numpy's log parts
    from glibc's."""
    keep = []
    tot = 0
    t0 = time.time()
    while sum(len(k) for k in keep) < want and tot < cap:
        x = adversarial(s, j, 1_100_000, lane)
        tot += len(x)
        m = np.log(x) != glibc_log(x)
        keep.append(x[m])
    x = np.concatenate(keep)
    return x, tot, time.time() - t0


def evaluate(x, steps, s, tag, report):
    ss = DEFAULT_S[steps] if s is None else s
    t = twin(x, steps, s)
    nat = native(x, steps, ss)
    L = glibc_log(x)
    Ln = np.log(x)
    logdiff = Ln != L
    for k in t:
        a, b = t[k], nat[k]
        u = ulps(a, b)
        key = f"{tag} steps={steps} s={ss} {k}"
        rec = {"n": int(len(x)), "log mismatches": int(logdiff.sum()),
               "lanes differ": int((a != b).sum()),
               "lanes differ where logs agree": int(((a != b) & ~logdiff).sum()),
               "max ulps": float(u.max())}
        if k != "sqrt":
            j = int(k[3:])
            f = np.sqrt(((2.0 * x) * ss) / np.cosh(np.pi * j * ss))
            Sc = f * (1.0 + np.abs(j * ss * L))
            K = np.abs(a - b) / (EPS * Sc)
            i = int(np.argmax(K))
            rec.update({"max K": float(K.max()), "K>1": int((K > 1).sum()),
                        "K>0.9": int((K > 0.9).sum()),
                        "argmax": {"x": float(x[i]), "L": float(L[i]),
                                   "js|L|": float(j * ss * abs(L[i])),
                                   "ulps": float(u[i])}})
        else:
            rec["max K"] = 0.0 if u.max() == 0 else None
        report[key] = rec
        print(key, json.dumps(rec), flush=True)


def main(out):
    report = {}
    t0 = time.time()
    N = int(os.environ.get("N", 200_000))
    R1 = rng.uniform(0.0, 1e3, N)
    R1 = R1[R1 > 0]
    R2 = np.exp(rng.uniform(math.log(1e-300), math.log(1e300), N))
    R2b = np.ldexp(rng.uniform(1.0, 2.0, N), rng.integers(-996, 997, N))
    R3 = 1.0 + rng.normal(0.0, 1e-3, N)
    for steps, s in CONFIGS:
        for tag, x in (("R1", R1), ("R2", R2), ("R2b", R2b), ("R3", R3)):
            evaluate(x, steps, s, tag, report)
        ss = DEFAULT_S[steps] if s is None else s
        for j in range(1, steps):
            for lane in ("cos", "sin"):
                xa = adversarial(ss, j, N, lane)
                evaluate(xa, steps, s, f"ADV(j={j},{lane})", report)
                xe, tot, secs = enriched(ss, j, lane, int(os.environ.get("ENR_WANT", 2000)), int(os.environ.get("ENR_CAP", 8_000_000)))
                report[f"ENR(j={j},{lane}) steps={steps} s={ss} candidates"] = {"candidates": tot, "kept": int(len(xe)), "secs": round(secs, 1)}
                print("enriched", steps, ss, j, lane, tot, len(xe), round(secs, 1), flush=True)
                evaluate(xe, steps, s, f"ENR(j={j},{lane})", report)
    report["secs"] = round(time.time() - t0, 1)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
