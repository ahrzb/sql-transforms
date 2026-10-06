"""E4: is the matvec K a constant? K = |twin - native| / (eps * S) against n,
for rows near a positive mean (the PCA cancellation), and a structured row
where the left-to-right order provably loses (n-2) half-ulps."""
import numpy as np, threadpoolctl
from fractions import Fraction as F
EPS = 2.0**-52; rng = np.random.default_rng(11)
print("BLAS", threadpoolctl.threadpool_info()[0]["architecture"])
def l2r(X, C):  # left to right, products rounded, no FMA: numpy accumulate is sequential
    P = X[:, None, :] * C[None, :, :]
    return np.add.accumulate(P, axis=2)[:, :, -1]
# sanity: accumulate == python loop
X = rng.normal(size=(3, 37)); C = rng.normal(size=(2, 37))
for r in range(3):
    for j in range(2):
        acc = X[r, 0] * C[j, 0]
        for i in range(1, 37): acc = acc + X[r, i] * C[j, i]
        assert acc == l2r(X, C)[r, j]
for kind in ["positive c", "mixed-sign c"]:
    for n in [4, 16, 32, 128, 512, 2048, 8192]:
        rows = 4000 if n <= 512 else 600
        m = rng.uniform(50, 150, n)
        C = rng.uniform(0, 1, (4, n)) if kind == "positive c" else rng.normal(size=(4, n))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        X = m + rng.normal(0, 1, (rows, n)) * 10.0 ** rng.uniform(-6, 0, (rows, 1))
        M = (m.reshape(1, -1) @ C.T)[0]
        T = np.vstack([np.asarray(x.reshape(1, -1) @ C.T)[0] for x in X]) - M   # row by row, as the twin
        N = l2r(X, C) - M
        S = np.abs(X[:, None, :] * C[None, :, :]).sum(axis=2) + np.abs(M)
        K = np.abs(T - N) / (EPS * S)
        print(f"{kind:12s} n={n:5d} lanes={K.size:6d} max K={K.max():7.3f} p99 K={np.percentile(K, 99):6.3f}"
              f"  max K/sqrt(n)={K.max() / np.sqrt(n):.3f}  derived bound n+2={n + 2}")
print("structured row x=[1, u, ..., u, -1], u=2^-53, c=1:")
u = 2.0**-53
for n in [8, 32, 128, 1024]:
    x = np.array([1.0] + [u] * (n - 2) + [-1.0]); C = np.ones((2, n))
    T = float((x.reshape(1, -1) @ C.T)[0, 0]); N = float(l2r(x.reshape(1, -1), C)[0, 0])
    ex = sum(F(v) for v in x); S = float(np.abs(x).sum())
    print(f"  n={n:5d} twin={T!r} native={N!r} exact={float(ex)!r}  K(twin,native)={abs(T - N) / (EPS * S):.2f}"
          f"  (n-2)/4={(n - 2) / 4:.2f}")
