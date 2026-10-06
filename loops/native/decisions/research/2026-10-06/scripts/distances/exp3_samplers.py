"""RBFSampler and SkewedChi2Sampler: twin (transform of a 1-row list) vs a
native-like SQL-order evaluation (left-to-right unfused dot, glibc log/cos via
math, constants as sklearn computes them).

Scales per lane j (c = output constant, L = log(x + s) or x):
  S_term = c * (sum_i |L_i W_ij| + |b_j| + 1)     (term scale of theta, through cos)
  S_mag  = c * (|theta_j| + 1)                     (magnitude of theta only)
  K = |native - twin| / (eps S)
"""
import math, struct, time
import numpy as np
from sklearn.kernel_approximation import RBFSampler, SkewedChi2Sampler

EPS = 2.0 ** -52


def ordered(x):
    (i,) = struct.unpack("<q", struct.pack("<d", x))
    return i if i >= 0 else -(i & 0x7FFF_FFFF_FFFF_FFFF)


def ulps(a, b):
    return abs(ordered(a) - ordered(b))


def dot_ltr_cols(L, W):
    # theta_j = L0*W0j + L1*W1j + ... left to right, for every j (vectorised over j; each op rounds as scalar)
    acc = L[0] * W[0]
    for i in range(1, len(L)):
        acc = acc + L[i] * W[i]
    return acc  # numpy elementwise ops are correctly rounded per element, unfused


def measure(label, est, rows, kind, s=None):
    W = np.asarray(est.random_weights_, dtype=np.float64)
    b = np.asarray(est.random_offset_, dtype=np.float64)
    D = W.shape[1]
    if kind == "rbf":
        c = (2.0 / est.n_components) ** 0.5
    else:
        c = float(np.sqrt(2.0) / np.sqrt(est.n_components))
    u_all, kt, km, cos_mis, log_mis, nl = [], [], [], 0, 0, 0
    for r in rows:
        x = [float(v) for v in r]
        twin = est.transform([x])[0]
        if kind == "rbf":
            L = np.array(x)
        else:
            z = [xi + s for xi in x]
            L = np.array([math.log(v) for v in z])
            log_mis += int((np.log(np.array(z)) != L).sum())
        th = dot_ltr_cols(L, W) + b
        nat = np.array([math.cos(t) for t in th]) * c
        # does numpy's cos equal glibc's on the twin's own theta?
        th_t = (np.asarray([L]) @ W)[0] + b
        cos_mis += int((np.cos(th_t) != np.array([math.cos(t) for t in th_t])).sum())
        St = c * (np.abs(L[:, None] * W).sum(axis=0) + np.abs(b) + 1)
        Sm = c * (np.abs(th) + 1)
        d = np.abs(nat - twin)
        kt.append(d / (EPS * St)); km.append(d / (EPS * Sm))
        u_all.extend(ulps(float(p), float(q)) for p, q in zip(nat, twin))
        nl += D
    kt = np.concatenate(kt); km = np.concatenate(km); u = np.array(u_all, dtype=float)
    print(f"{label:<52} lanes={nl:>6} ulp max={u.max():.3g} p99={np.percentile(u,99):.3g} | "
          f"K_term max={kt.max():.3f} p99={np.percentile(kt,99):.3f} | K_mag max={km.max():.3g} | "
          f"np.cos!=glibc {cos_mis} | np.log!=glibc {log_mis}", flush=True)
    return kt.max(), u.max()


def main():
    t0 = time.time()
    rng = np.random.default_rng(5)
    worst = []
    for nf in (4, 16, 64):
        for gamma in (0.1, 1.0, "scale"):
            for data in ("std", "offset100"):
                X = rng.normal(size=(1000, nf))
                if data == "offset100":
                    X = X + 100.0
                est = RBFSampler(gamma=gamma, n_components=100, random_state=nf).fit(X)
                rows = X[rng.choice(1000, 120, replace=False)]
                worst.append(measure(f"RBFSampler nf={nf} gamma={gamma} {data}", est, rows, "rbf"))
    for nf in (4, 16, 64):
        for s in (1.0, 0.01):
            for data in ("unif0-10", "lognormal", "sparsehist"):
                if data == "unif0-10":
                    X = rng.uniform(0, 10, size=(1000, nf))
                elif data == "lognormal":
                    X = rng.lognormal(0, 2, size=(1000, nf))
                else:
                    X = rng.poisson(0.7, size=(1000, nf)).astype(float)
                est = SkewedChi2Sampler(skewedness=s, n_components=100, random_state=nf).fit(X)
                rows = X[rng.choice(1000, 120, replace=False)]
                worst.append(measure(f"SkewedChi2 nf={nf} s={s} {data}", est, rows, "skc", s))
    print(f"POOLED max K_term={max(w[0] for w in worst):.3f}  max ulps={max(w[1] for w in worst):.3g}")
    print(f"elapsed {time.time()-t0:.1f}s")


main()
