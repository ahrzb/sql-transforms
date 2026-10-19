"""K vs n. Errors of the dot x.c in units of eps * S0, S0 = sum_i |x_i c_i|.
exact = correctly rounded x.c (Dekker TwoProduct pieces, math.fsum per lane).
native = left-to-right, products rounded, unfused (the entry's `dot`)."""

import math
import pickle
import struct
import sys

import numpy as np

EPS = 2.0**-52
D = sys.argv[1]
data = pickle.load(open(f"{D}/kvn_data.pkl", "rb"))
cfgs = ["SkylakeX_t1", "SkylakeX_t4", "Haswell_t1", "Haswell_t4", "Sandybridge_t1", "Nehalem_t1", "Prescott_t1"]
tw = {c: pickle.load(open(f"{D}/kvn_{c}.pkl", "rb")) for c in cfgs}


def split(a):
    t = 134217729.0 * a
    h = t - (t - a)
    return h, a - h


def ordered(x):
    (i,) = struct.unpack("<q", struct.pack("<d", x))
    return i if i >= 0 else -(i & 0x7FFF_FFFF_FFFF_FFFF)


def ulps(a, b):
    return abs(ordered(float(a)) - ordered(float(b)))


# sanity: Dekker error term equals fma's
a, b = np.random.default_rng(0).standard_normal((2, 1000)) * 1e3
p = a * b
ah, al = split(a)
bh, bl = split(b)
e = al * bl - (((p - ah * bh) - al * bh) - ah * bl)
assert all(math.fma(x, y, -z) == w for x, y, z, w in zip(a, b, p, e))


def q(v, pct):
    return float(np.percentile(v, pct))


rows_out = []
for (kind, n), d in data.items():
    C, X, mean = d["C"], d["X"], d["mean"]
    Rr, k = X.shape[0], C.shape[0]
    exact = np.empty((Rr, k))
    S0 = np.empty((Rr, k))
    ch, cl = split(C)
    for r in range(Rr):
        x = X[r]
        P = x[None, :] * C
        xh, xl = split(x)
        E = xl[None, :] * cl - (((P - xh[None, :] * ch) - xl[None, :] * ch) - xh[None, :] * cl)
        S0[r] = np.abs(P).sum(axis=1)
        for j in range(k):
            exact[r, j] = math.fsum(np.concatenate([P[j], E[j]]).tolist())
    ltr = X[:, 0:1] * C[:, 0][None, :]
    for i in range(1, n):
        ltr = ltr + X[:, i : i + 1] * C[:, i][None, :]
    u = EPS * S0
    ref = tw["SkylakeX_t1"][(kind, n)]
    K = {
        "LTR-exact": np.abs(ltr - exact) / u,
        "gemv(SKX)-exact": np.abs(ref["row"] - exact) / u,
        "LTR-gemv(SKX)": np.abs(ltr - ref["row"]) / u,
        "gemm(SKX)-gemv(SKX)": np.abs(ref["batch"] - ref["row"]) / u,
        "gemv(HSW)-gemv(SKX)": np.abs(tw["Haswell_t1"][(kind, n)]["row"] - ref["row"]) / u,
        "gemv(SNB)-gemv(SKX)": np.abs(tw["Sandybridge_t1"][(kind, n)]["row"] - ref["row"]) / u,
        "gemv(PSC)-gemv(SKX)": np.abs(tw["Prescott_t1"][(kind, n)]["row"] - ref["row"]) / u,
        "gemv(SKX,4thr)-gemv(SKX,1thr)": np.abs(tw["SkylakeX_t4"][(kind, n)]["row"] - ref["row"]) / u,
        "gemm(SKX,4thr)-gemm(SKX,1thr)": np.abs(tw["SkylakeX_t4"][(kind, n)]["batch"] - ref["batch"]) / u,
        "gemm(HSW)-gemm(SKX)": np.abs(tw["Haswell_t1"][(kind, n)]["batch"] - ref["batch"]) / u,
    }
    for name, v in K.items():
        rows_out.append((kind, n, name, q(v, 50), q(v, 99), float(v.max())))
    # whole PCA lanes
    if "est" in d:
        est = d["est"]
        s = np.sqrt(est.explained_variance_)
        s[s < EPS] = EPS
        mc = ref["mc"]
        native = (ltr - mc) / s
        Sm = np.abs(mean[None, :] * C).sum(axis=1)
        S2 = (S0 + Sm[None, :]) / s
        Srec = (S0 + np.abs(mc)[None, :]) / s
        for name, a_, b_ in [
            ("lane: native-twin(SKX row)", native, ref["lane_row"]),
            ("lane: twin batch-row (SKX)", ref["lane_batch"], ref["lane_row"]),
            ("lane: twin SNB row-SKX row", tw["Sandybridge_t1"][(kind, n)]["lane_row"], ref["lane_row"]),
            ("lane: twin HSW row-SKX row", tw["Haswell_t1"][(kind, n)]["lane_row"], ref["lane_row"]),
            ("lane: native-twin(SNB row)", native, tw["Sandybridge_t1"][(kind, n)]["lane_row"]),
        ]:
            dd = np.abs(a_ - b_)
            uu = np.array([ulps(p_, q_) for p_, q_ in zip(a_.ravel(), b_.ravel())])
            rows_out.append((kind, n, name + " [eps*S2]", q(dd / (EPS * S2), 50), q(dd / (EPS * S2), 99), float((dd / (EPS * S2)).max())))
            rows_out.append((kind, n, name + " [eps*S_rec]", q(dd / (EPS * Srec), 50), q(dd / (EPS * Srec), 99), float((dd / (EPS * Srec)).max())))
            rows_out.append((kind, n, name + " [ulps]", q(uu, 50), q(uu, 99), float(uu.max())))

print(f"{'data':7s} {'n':>5s} {'quantity':45s} {'p50':>9s} {'p99':>9s} {'max':>12s}")
for kind, n, name, a50, a99, amax in rows_out:
    print(f"{kind:7s} {n:5d} {name:45s} {a50:9.3f} {a99:9.3f} {amax:12.4g}")
print("\nprovable K (|native - twin| <= K eps S0): any order on the twin side: n;"
      " SkylakeX gemv_t (depth n/4+3): (n + n/4 + 3)/2")
for n in [4, 32, 128, 512, 2048]:
    print(n, "any:", n, " skx:", (n + n / 4 + 3) / 2)
