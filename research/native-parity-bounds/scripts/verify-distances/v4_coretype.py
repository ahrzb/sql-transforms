"""Twin across OpenBLAS kernels: same fitted state + rows, OPENBLAS_CORETYPE per subprocess."""
import os, sys, pickle, subprocess, math
from fractions import Fraction
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)) + "/v4data"
EPS = 2.0 ** -52
if len(sys.argv) > 1 and sys.argv[1] == "child":
    from threadpoolctl import threadpool_info
    arch = [i.get("architecture") for i in threadpool_info() if i.get("internal_api") == "openblas"]
    st = pickle.load(open(f"{D}/state.pkl", "rb"))
    out = {"arch": arch}
    for key, (est, rows) in st.items():
        out[key] = np.vstack([est.transform([list(map(float, r))])[0] for r in rows])
    pickle.dump(out, open(f"{D}/out_{sys.argv[2]}.pkl", "wb"))
    sys.exit(0)
os.makedirs(D, exist_ok=True)
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import RBFSampler, SkewedChi2Sampler
rng = np.random.default_rng(1234)
st = {}
for n in (6, 24, 96):
    X = rng.normal(size=(1200, n)) * 2 + rng.normal(size=(1, n)) * 15
    km = KMeans(n_clusters=6, n_init=1, random_state=0).fit(X)
    C = km.cluster_centers_
    rows = np.vstack([X[:400], C[rng.integers(0, 6, 400)] * (1 + 1e-8 * rng.normal(size=(400, n)))])
    st[f"km{n}"] = (km, rows)
    rb = RBFSampler(gamma=0.5, n_components=200, random_state=n).fit(X)
    st[f"rbf{n}"] = (rb, X[400:700])
    Xp = np.abs(X)
    sk = SkewedChi2Sampler(skewedness=0.5, n_components=200, random_state=n).fit(Xp)
    st[f"sk{n}"] = (sk, Xp[700:1000])
pickle.dump(st, open(f"{D}/state.pkl", "wb"))
cores = ["SkylakeX", "Haswell", "Zen", "Sandybridge", "Nehalem", "Prescott", "Cooperlake"]
res = {}
for c in cores:
    env = dict(os.environ, OPENBLAS_CORETYPE=c, OPENBLAS_NUM_THREADS="1")
    subprocess.run([sys.executable, __file__, "child", c], env=env, check=True)
    res[c] = pickle.load(open(f"{D}/out_{c}.pkl", "rb"))
    print(c, "->", res[c]["arch"])
def ordered(v):
    i = np.asarray(v, dtype=np.float64).view(np.int64)
    return np.where(i >= 0, i, -(i & 0x7FFFFFFFFFFFFFFF))
base = res["SkylakeX"]
for c in cores[1:]:
    for key, (est, rows) in st.items():
        a, b = base[key], res[c][key]
        nd = int((a != b).sum()); ul = int(np.abs(ordered(a).astype(object) - ordered(b).astype(object)).max()) if nd else 0
        if key.startswith("km"):
            C = est.cluster_centers_
            Ks = []
            for i, r in enumerate(rows):
                for k in range(len(C)):
                    if a[i, k] == b[i, k]: continue
                    S2 = float(sum(Fraction(v) ** 2 for v in r) + sum(Fraction(v) ** 2 for v in C[k]) + 2 * sum(abs(Fraction(p) * Fraction(q)) for p, q in zip(r, C[k])))
                    Ks.append(abs(float(Fraction(float(a[i, k])) ** 2 - Fraction(float(b[i, k])) ** 2)) / (EPS * S2))
            Kmax = max(Ks) if Ks else 0
        else:
            W, bb = est.random_weights_, est.random_offset_
            L = rows if key.startswith("rbf") else np.log(rows + est.skewedness)
            cst = (2.0 / est.n_components) ** 0.5
            S = cst * (np.abs(L[:, :, None] * W[None, :, :]).sum(axis=1) + np.abs(bb)[None, :] + 1)
            Kmax = float((np.abs(a - b) / (EPS * S)).max())
        print(f"SkylakeX vs {c:<12} {key:<6} differ {nd:>6}/{a.size:<6} ({100*nd/a.size:4.1f}%) max ulps {ul:.3g}  K max {Kmax:.3f}")
