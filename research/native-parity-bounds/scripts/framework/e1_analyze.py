"""E1 step 3: native (left-to-right, no FMA, as the SQL entry sums) and an
exact reference (Fractions), against the twin under each OpenBLAS kernel.
K = |a-b| / (eps*S): S = record's term scale (sum|x_i c_i| + |M|)/scale, with
M the double the twin computes; S_full uses sum|m_i c_i| in place of |M|."""
import math, pickle, sys, numpy as np
from fractions import Fraction as F
EPS = 2.0**-52
cases = pickle.load(open(sys.argv[1], "rb"))
variants = sys.argv[2].split(",")
tw = {v: np.load(f"twin_{v}.npz") for v in variants}

def ordered(x):
    i = np.asarray(x, dtype=np.float64).view(np.int64).astype(object)
    return np.array([v if v >= 0 else -(v & 0x7FFF_FFFF_FFFF_FFFF) for v in i], dtype=object)

nat = {v: [] for v in variants}; exact = []; S = {v: [] for v in variants}; Sfull = []; nfeat = []
for v in variants: tw[v] = (tw[v]["lanes"], tw[v]["M"])
off = 0
for seed, est, x in cases:
    C = est.components_; k, n = C.shape
    sc = np.sqrt(est.explained_variance_); sc[sc < np.finfo(float).eps] = np.finfo(float).eps
    for j in range(k):
        c = [float(t) for t in C[j]]
        acc = x[0] * c[0]
        for i in range(1, n):
            acc = acc + x[i] * c[i]           # left to right, each op rounded
        A = sum(abs(x[i] * c[i]) for i in range(n))
        for v in variants:
            M = float(tw[v][1][off + j])
            nat[v].append((acc - M) / float(sc[j]))
            S[v].append((A + abs(M)) / float(sc[j]))
        m = [float(t) for t in est.mean_]
        ex = (sum(F(x[i]) * F(c[i]) for i in range(n)) - sum(F(m[i]) * F(c[i]) for i in range(n))) / F(float(sc[j]))
        exact.append(ex)
        Sfull.append((A + sum(abs(m[i] * c[i]) for i in range(n))) / float(sc[j]))
        nfeat.append(n)
    off += k
nfeat = np.array(nfeat)

def cmp(a, b, s, name):
    a = np.asarray(a, float); b = np.asarray(b, float); s = np.asarray(s, float)
    fin = np.isfinite(a) & np.isfinite(b) & np.isfinite(s) & (s > 0)
    nonfin_mismatch = int(np.sum(~(np.isfinite(a) & np.isfinite(b)) & ~((a == b) | (np.isnan(a) & np.isnan(b)))))
    a, b, s = a[fin], b[fin], s[fin]
    d = np.abs(a - b)
    ul = np.abs(ordered(a) - ordered(b)).astype(float)
    K = d / (EPS * s)
    flips = int(np.sum(np.sign(a) != np.sign(b)))
    return dict(pair=name, lanes=int(fin.sum()), differ=int(np.sum(a != b)), max_ulps=float(ul.max()),
                p99_ulps=float(np.percentile(ul, 99)), p99_K=float(np.percentile(K, 99)), max_K=float(K.max()),
                sign_flips=flips, nonfinite_mismatch=nonfin_mismatch)

def vs_exact(a, name):
    a = np.asarray(a, float); s = np.asarray(Sfull)
    fin = np.isfinite(a) & np.isfinite(s) & (s > 0)
    K = np.array([abs(float(F(float(ai)) - e)) / (EPS * si) if f else 0.0 for ai, e, si, f in zip(a, exact, s, fin)])
    Kf = K[fin]
    worstn = int(nfeat[fin][np.argmax(Kf)])
    return dict(pair=name + " vs exact (S_full)", lanes=int(fin.sum()), p99_K=float(np.percentile(Kf, 99)),
                max_K=float(Kf.max()), n_at_max=worstn, derived_Kdet_at_that_n=(worstn + 2) / 2)

rows = []
base = variants[0]
for v in variants:
    rows.append(cmp(tw[v][0], nat[v], S[v], f"twin[{v}] vs native(M_{v})"))
for i, v in enumerate(variants):
    for w in variants[i + 1:]:
        rows.append(cmp(tw[v][0], tw[w][0], Sfull, f"twin[{v}] vs twin[{w}] (S_full)"))
rows.append(cmp(tw[variants[1]][0], nat[base], Sfull, f"twin[{variants[1]}] vs native(M_{base}) (S_full)"))
for v in variants:
    rows.append(vs_exact(tw[v][0], f"twin[{v}]"))
rows.append(vs_exact(nat[base], "native"))
# M itself across kernels
Ms = np.array([tw[v][1] for v in variants])
Md = np.max(np.abs(Ms - Ms[0]), axis=0)
print("M differs across kernels on", int(np.sum(Md > 0)), "of", Ms.shape[1], "projected-mean constants")
# K vs n for native vs base twin
a = np.asarray(tw[base][0]); b = np.asarray(nat[base]); s = np.asarray(S[base])
fin = np.isfinite(a) & np.isfinite(b) & (s > 0)
K = np.abs(a - b) / (EPS * s)
for lo, hi in [(1, 4), (5, 16), (17, 32)]:
    m = fin & (nfeat >= lo) & (nfeat <= hi)
    if m.any():
        print(f"n in [{lo},{hi}]: lanes {int(m.sum())}, max K {K[m].max():.3f}, p99 K {np.percentile(K[m], 99):.3f}")
for r in rows:
    print(r)
