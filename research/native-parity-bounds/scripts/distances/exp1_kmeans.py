"""Distances to fitted centres: twin (sklearn transform on a 1-row list, as
PythonTransform calls it) vs a native-like SQL-order evaluation.

native-like lane k:  q = ((-2*dot_ltr(x, c_k)) + row_sumsq(x)) + CC_k
                     d = sqrt(max(q, 0))
  dot_ltr  : x0*c0 + x1*c1 + ... left to right, unfused (SQL's order)
  row_sumsq: the repo's bit-exact spelling of numpy's einsum row_norms
  CC_k     : row_norms(C, squared=True)[k], a numpy constant (as the twin)
direct-native lane k: sqrt(sum_ltr (x_i - c_ki)^2)   (the accurate formula)

Scales:  S2 = ||x||^2 + ||c_k||^2 + 2 sum_i |x_i c_ki|   (term scale of d^2)
  K2 = |d_n^2 - d_t^2| / (eps S2)      (squared-lane bound, exact arithmetic)
  K1 = |d_n - d_t| / (eps sqrt(S2))    (a linear bound on the lane itself)
"""

import math
import struct
import sys
import time
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
from sklearn.cluster import Birch, BisectingKMeans, KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.utils.extmath import row_norms

from rowsumsq import row_sumsq, row_sumsq_is_numpys  # copy of the repo's _helpers.row_sumsq

getcontext().prec = 60
EPS = 2.0**-52


def ordered(x):
    (i,) = struct.unpack("<q", struct.pack("<d", x))
    return i if i >= 0 else -(i & 0x7FFF_FFFF_FFFF_FFFF)


def ulps(a, b):
    return abs(ordered(a) - ordered(b))


def dot_ltr(x, c):
    acc = x[0] * c[0]
    for i in range(1, len(x)):
        acc = acc + x[i] * c[i]
    return acc


def native_lanes(x, C, CC):
    xx = row_sumsq(x)
    out = []
    for k in range(len(C)):
        q = (-2.0 * dot_ltr(x, C[k]) + xx) + CC[k]
        out.append(math.sqrt(max(q, 0.0)))
    return out


def direct_lanes(x, C):
    out = []
    for c in C:
        acc = 0.0
        for xi, ci in zip(x, c):
            t = xi - ci
            acc = acc + t * t
        out.append(math.sqrt(acc))
    return out


def exact_q(x, c):
    return sum((Fraction(a) - Fraction(b)) ** 2 for a, b in zip(x, c))


def fsqrt(fr):
    if fr == 0:
        return 0.0
    return float((Decimal(fr.numerator) / Decimal(fr.denominator)).sqrt())


def S2_of(x, c):
    return math.fsum(v * v for v in x) + math.fsum(v * v for v in c) + 2 * math.fsum(
        abs(a * b) for a, b in zip(x, c)
    )


def sqdiff(a, b):  # |a^2 - b^2| exactly, as a float
    return abs(float(Fraction(a) ** 2 - Fraction(b) ** 2))


def run(est, rows, label, stats_out):
    C = est.cluster_centers_ if hasattr(est, "cluster_centers_") else est.subcluster_centers_
    C = np.asarray(C, dtype=np.float64)
    CC = row_norms(C, squared=True)
    Cl = [list(map(float, c)) for c in C]
    # sanity: CC per row equals CC of a one-row C
    for k in range(len(C)):
        assert CC[k] == row_norms(C[k : k + 1], squared=True)[0]
    rec = dict(ulp=[], K2=[], K1=[], K2_twin_exact=[], K2_direct=[], K2_native_exact=[],
               ulp_twin_exact=[], ulp_direct=[], tz_nnz=0, nz_tnz=0, lanes=0, reimpl_bad=0, xx_bad=0)
    for r in rows:
        x = [float(v) for v in r]
        twin = est.transform([x])[0]
        # reimplementation of the twin with numpy, bit-exact check
        X = np.asarray([x])
        re = np.sqrt(np.maximum((-2 * (X @ C.T) + row_norms(X, squared=True)[:, None]) + CC[None, :], 0))[0]
        rec["reimpl_bad"] += int(not np.array_equal(re, twin))
        rec["xx_bad"] += int(row_sumsq(x) != float(row_norms(X, squared=True)[0]))
        nat = native_lanes(x, Cl, CC)
        dirr = direct_lanes(x, Cl)
        for k in range(len(C)):
            t, n, dd = float(twin[k]), nat[k], dirr[k]
            s2 = S2_of(x, Cl[k])
            qe = exact_q(x, Cl[k])
            de = fsqrt(qe)
            rec["lanes"] += 1
            rec["ulp"].append(ulps(n, t))
            rec["K2"].append(sqdiff(n, t) / (EPS * s2))
            rec["K1"].append(abs(n - t) / (EPS * math.sqrt(s2)))
            rec["K2_direct"].append(sqdiff(dd, t) / (EPS * s2))
            rec["ulp_direct"].append(ulps(dd, t))
            rec["K2_twin_exact"].append(abs(float(Fraction(t) ** 2 - qe)) / (EPS * s2))
            rec["K2_native_exact"].append(abs(float(Fraction(n) ** 2 - qe)) / (EPS * s2))
            rec["ulp_twin_exact"].append(ulps(t, de))
            if t == 0.0 and n != 0.0:
                rec["tz_nnz"] += 1
            if n == 0.0 and t != 0.0:
                rec["nz_tnz"] += 1
    a = {k: np.asarray(v, dtype=float) for k, v in rec.items() if isinstance(v, list)}
    line = (
        f"{label:<44} lanes={rec['lanes']:>6} "
        f"ulp max={a['ulp'].max():.3g} p99={np.percentile(a['ulp'], 99):.3g} | "
        f"K2 max={a['K2'].max():.3f} p99={np.percentile(a['K2'], 99):.3f} | "
        f"K1 max={a['K1'].max():.3g} | "
        f"twin-vs-exact K2 max={a['K2_twin_exact'].max():.3f} ulp max={a['ulp_twin_exact'].max():.3g} | "
        f"native-vs-exact K2 max={a['K2_native_exact'].max():.3f} | "
        f"direct-vs-twin K2 max={a['K2_direct'].max():.3f} ulp max={a['ulp_direct'].max():.3g} | "
        f"twin0/native>0={rec['tz_nnz']} native0/twin>0={rec['nz_tnz']} reimpl_mismatch_rows={rec['reimpl_bad']} xx_mismatch_rows={rec['xx_bad']}"
    )
    print(line, flush=True)
    stats_out.append((label, rec["lanes"], a, rec))


def main():
    t0 = time.time()
    print("row_sumsq_is_numpys:", row_sumsq_is_numpys())
    rng = np.random.default_rng(7)
    stats = []
    for nf in (2, 8, 32):
        for kind in ("centered", "offset1e3", "mixedscale"):
            X, _ = make_blobs(n_samples=2000, n_features=nf, centers=5, cluster_std=1.0,
                              center_box=(-10, 10), random_state=nf)
            if kind == "offset1e3":
                X = X + 1e3
            elif kind == "mixedscale":
                X = X * (10.0 ** (np.arange(nf) % 5))[None, :]
            est = KMeans(n_clusters=5, n_init=1, random_state=0).fit(X)
            C = est.cluster_centers_
            sd = X.std(axis=0)
            far = X[rng.choice(len(X), 300, replace=False)] + rng.normal(size=(300, nf)) * sd * 0.1
            run(est, far, f"KMeans nf={nf} {kind} far", stats)
            for delta in (1e-2, 1e-5, 1e-8, 1e-11):
                ks = rng.integers(0, len(C), 150)
                near = C[ks] + delta * rng.normal(size=(150, nf)) * sd
                run(est, near, f"KMeans nf={nf} {kind} near d={delta:g}", stats)
            run(est, C.copy(), f"KMeans nf={nf} {kind} x=centre", stats)
    # the other estimators: same code path, one config each
    X, _ = make_blobs(n_samples=2000, n_features=8, centers=5, random_state=3)
    X = X + 1e2
    for est in (MiniBatchKMeans(n_clusters=5, n_init=1, random_state=0),
                BisectingKMeans(n_clusters=5, random_state=0),
                Birch(threshold=2.0, n_clusters=None)):
        est.fit(X)
        C = np.asarray(getattr(est, "cluster_centers_", getattr(est, "subcluster_centers_", None)))
        sd = X.std(axis=0)
        ks = rng.integers(0, len(C), 200)
        rows = np.vstack([X[rng.choice(len(X), 200, replace=False)],
                          C[ks] + 1e-6 * rng.normal(size=(200, 8)) * sd])
        run(est, rows, f"{type(est).__name__} nf=8 offset1e2 (k={len(C)})", stats)
    # pooled
    allK2 = np.concatenate([s[2]["K2"] for s in stats])
    allulp = np.concatenate([s[2]["ulp"] for s in stats])
    allK2t = np.concatenate([s[2]["K2_twin_exact"] for s in stats])
    allK2d = np.concatenate([s[2]["K2_direct"] for s in stats])
    print(f"POOLED lanes={allK2.size} K2 max={allK2.max():.3f} p99={np.percentile(allK2,99):.3f} "
          f"p99.9={np.percentile(allK2,99.9):.3f} | ulp max={allulp.max():.4g} | "
          f"twin-vs-exact K2 max={allK2t.max():.3f} | direct-vs-twin K2 max={allK2d.max():.3f}")
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
