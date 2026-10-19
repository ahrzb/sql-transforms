"""E1c: how far the twin itself is from the correctly rounded lane, in ulps of
the result (PCA whiten, the catalog's fixtures, seeds 0-199)."""
import math, pickle, numpy as np
from fractions import Fraction as F
cases = pickle.load(open("pca_cases.pkl", "rb")); tw = np.load("twin_default.npz")["lanes"]
def ordered(v):
    import struct
    (i,) = struct.unpack("<q", struct.pack("<d", v)); return i if i >= 0 else -(i & 0x7FFF_FFFF_FFFF_FFFF)
off = 0; worst = []; 
for seed, est, x in cases:
    C = est.components_; k, n = C.shape
    sc = np.sqrt(est.explained_variance_); sc[sc < np.finfo(float).eps] = np.finfo(float).eps
    for j in range(k):
        ex = (sum(F(x[i]) * F(float(C[j, i])) for i in range(n)) - sum(F(float(est.mean_[i])) * F(float(C[j, i])) for i in range(n))) / F(float(sc[j]))
        t = float(tw[off + j])
        if math.isfinite(t):
            try: r = float(ex)
            except OverflowError: r = math.inf
            if math.isfinite(r): worst.append(abs(ordered(t) - ordered(r)))
    off += k
w = np.array(worst, dtype=float)
print(f"twin vs correctly-rounded lane: lanes {len(w)}, exact {int((w == 0).sum())}, max ulps {w.max():.0f}, p99 {np.percentile(w, 99):.0f}, p99.9 {np.percentile(w, 99.9):.0f}")
