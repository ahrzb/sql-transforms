"""K = |ltr - gemv| / (eps * sum|x c|) vs n, all-positive (uniform(0,1)) and mixed-sign (N(0,1)) data.
Fixed lane budget per n (so max is comparable across n). Twin: 1-row x @ C.T (dgemv_t) and x @ W (dgemv_n)."""
import numpy as np, math
EPS = 2.0 ** -52
rng = np.random.default_rng(8086)
print(f"{'data':<10}{'n':>6} {'lanes':>8} | gemv_t max  p99.9  p50 | gemv_n max p99.9 | exact-vs-ltr max (fsum ref)")
for kind in ("positive", "mixed"):
    for n in (8, 32, 128, 512, 2048):
        m = 64; lanes_target = 200000
        rows = max(1, lanes_target // m)
        C = rng.uniform(0, 1, size=(m, n)) if kind == "positive" else rng.normal(size=(m, n))
        W = np.ascontiguousarray(C.T)
        Kt, Kn, Ke = [], [], []
        for _ in range(rows):
            x = rng.uniform(0, 1, size=n) if kind == "positive" else rng.normal(size=n)
            tt = (x[None, :] @ C.T)[0]; tn = (x[None, :] @ W)[0]
            acc = x[0] * C[:, 0]
            for i in range(1, n):
                acc = acc + x[i] * C[:, i]
            S = np.array([math.fsum(v) for v in np.abs(x[None, :] * C)])
            ex = np.array([math.fsum(v) for v in (x[None, :] * C)])  # products rounded; fine as reference of sums
            Kt.append(np.abs(acc - tt) / (EPS * S)); Kn.append(np.abs(acc - tn) / (EPS * S)); Ke.append(np.abs(acc - ex) / (EPS * S))
        Kt, Kn, Ke = map(np.concatenate, (Kt, Kn, Ke))
        print(f"{kind:<10}{n:>6} {Kt.size:>8} | {Kt.max():9.3f} {np.percentile(Kt,99.9):6.3f} {np.median(Kt):5.3f} | {Kn.max():9.3f} {np.percentile(Kn,99.9):6.3f} | {Ke.max():.3f}", flush=True)
