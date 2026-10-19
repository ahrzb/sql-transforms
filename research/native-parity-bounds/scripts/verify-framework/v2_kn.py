"""Independent: K = |entry - twin|/(eps*S) on REAL PCA(whiten=True) fits of wide
positive data (twin = est.transform([x])[0] row by row, as PythonTransform), and the
structured row x=[1, u..u, -1]."""
import math, sys, numpy as np, threadpoolctl, warnings
from fractions import Fraction as F
from sklearn.decomposition import PCA
warnings.simplefilter("ignore")
EPS = 2.0 ** -52
print("BLAS", threadpoolctl.threadpool_info()[0]["architecture"])
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
def l2r(x, c):
    acc = x[0] * c[0]
    for i in range(1, len(x)): acc = acc + x[i] * c[i]
    return acc
for n in [8, 32, 64, 128, 256, 512, 1024]:
    # positive, correlated data (one dominant factor): PCA's first component is all-positive
    f = rng.normal(size=(4 * n, 1)); base = rng.uniform(50, 150, n)
    Xfit = base + 10 * f * rng.uniform(0.5, 1.5, n) + rng.normal(size=(4 * n, n))
    est = PCA(n_components=min(8, n), whiten=True).fit(Xfit)
    C = est.components_; sc = np.sqrt(est.explained_variance_)
    M = (est.mean_.reshape(1, -1) @ C.T)[0]
    rows = 400 if n <= 256 else 120
    X = est.mean_ + rng.normal(size=(rows, n)) * 10.0 ** rng.uniform(-6, 0, (rows, 1))
    Ks = []; Kpos = []
    for x in X:
        t = est.transform(x.reshape(1, -1))[0]
        xl = x.tolist()
        for j in range(C.shape[0]):
            c = C[j].tolist()
            e = (l2r(xl, c) - float(M[j])) / float(sc[j])
            S = (math.fsum(abs(a * b) for a, b in zip(xl, c)) + abs(float(M[j]))) / float(sc[j])
            K = abs(e - float(t[j])) / (EPS * S); Ks.append(K)
            if (C[j] > 0).all() or (C[j] < 0).all(): Kpos.append(K)
    Ks = np.array(Ks)
    print(f"real PCA n={n:5d}: lanes {Ks.size}, max K {Ks.max():.2f}, p99 K {np.percentile(Ks,99):.2f}, same-sign-component lanes {len(Kpos)} max K {max(Kpos) if Kpos else float('nan'):.2f}, max K/sqrt(n) {Ks.max()/math.sqrt(n):.3f}")
u = 2.0 ** -53
for n in [8, 27, 32, 128, 1024]:
    x = np.array([1.0] + [u] * (n - 2) + [-1.0]); C = np.ones((3, n))
    T = float((x.reshape(1, -1) @ C.T)[0, 0]); N = l2r(x.tolist(), [1.0] * n)
    ex = float(sum(F(v) for v in x)); S = math.fsum(abs(v) for v in x)
    print(f"structured n={n}: twin={T!r} ({T/ex if ex else 0:.3f} of exact) entry={N!r} exact={ex!r} K={abs(T-N)/(EPS*S):.3f} (n-2)/8={(n-2)/8:.3f}")
    # also as a PCA-like call: x @ C.T through sklearn-like path with different k
    for k in [1, 2, 8]:
        T2 = float((x.reshape(1, -1) @ np.ones((k, n)).T)[0, 0])
        print(f"    k={k}: twin={T2!r}")
