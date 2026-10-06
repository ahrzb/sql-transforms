"""Plain dot product, twin = numpy 1-row matmul (BLAS gemv: x @ C.T is the dgemv_t
path KMeans uses, x @ W the dgemv_n path RBFSampler uses), native = left to right
unfused.  K = |native - twin| / (eps * sum|x_i c_i|).  Inputs N(0,1)+offset 3."""
import numpy as np, time
EPS = 2.0 ** -52
rng = np.random.default_rng(31)
t0 = time.time()
print(f"{'n':>5} {'lanes':>7} | gemv_t K max  p99.9 | gemv_n K max  p99.9 | worst-case n")
for n in (8, 32, 128, 512, 2048):
    m = 64
    C = rng.normal(size=(m, n)) + 3.0
    W = np.ascontiguousarray(C.T)
    rows = max(50, 400000 // (n * m))
    Kt, Kn = [], []
    for _ in range(rows):
        x = rng.normal(size=n) + 3.0
        X = x[None, :]
        tt = (X @ C.T)[0]; tn = (X @ W)[0]
        acc = x[0] * C[:, 0]
        for i in range(1, n):
            acc = acc + x[i] * C[:, i]
        S = np.abs(x[None, :] * C).sum(axis=1)
        Kt.append(np.abs(acc - tt) / (EPS * S)); Kn.append(np.abs(acc - tn) / (EPS * S))
    Kt = np.concatenate(Kt); Kn = np.concatenate(Kn)
    print(f"{n:>5} {Kt.size:>7} | {Kt.max():11.3f} {np.percentile(Kt, 99.9):6.3f} | {Kn.max():11.3f} {np.percentile(Kn, 99.9):6.3f} | {n}")
print(f"elapsed {time.time()-t0:.1f}s")
