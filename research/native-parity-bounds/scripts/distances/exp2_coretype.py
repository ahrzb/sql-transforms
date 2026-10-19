"""The twin under different OpenBLAS kernels (OPENBLAS_CORETYPE set before
import, one subprocess each): KMeans.transform and RBFSampler.transform on
the same fitted state and rows, one row per call (as PythonTransform).

Usage: exp2_coretype.py prep | run CORETYPE | compare
"""
import math, os, struct, subprocess, sys
from fractions import Fraction
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EPS = 2.0 ** -52
CORES = ["SkylakeX", "Haswell", "Sandybridge", "Prescott", "Zen", "Cooperlake"]


def ordered(x):
    (i,) = struct.unpack("<q", struct.pack("<d", x))
    return i if i >= 0 else -(i & 0x7FFF_FFFF_FFFF_FFFF)


def prep():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.kernel_approximation import RBFSampler
    import pickle
    rng = np.random.default_rng(11)
    out = {}
    for nf in (8, 32, 64):
        X, _ = make_blobs(n_samples=3000, n_features=nf, centers=8, random_state=nf)
        X = X + 50.0
        km = KMeans(n_clusters=8, n_init=1, random_state=0).fit(X)
        C = km.cluster_centers_
        rows = np.vstack([X[rng.choice(len(X), 300, replace=False)],
                          C[rng.integers(0, 8, 300)] + 1e-7 * rng.normal(size=(300, nf))])
        rbf = RBFSampler(gamma=0.5, n_components=100, random_state=0).fit((X - 50) / 5)
        rrows = (rows - 50) / 5
        out[nf] = (km, rows, rbf, rrows)
    with open(os.path.join(D, "coretype.pkl"), "wb") as f:
        pickle.dump(out, f)


def run(core):
    import pickle
    import numpy  # noqa
    from threadpoolctl import threadpool_info
    arch = [i.get("architecture") for i in threadpool_info() if i.get("internal_api") == "openblas"]
    with open(os.path.join(D, "coretype.pkl"), "rb") as f:
        out = pickle.load(f)
    res = {}
    for nf, (km, rows, rbf, rrows) in out.items():
        a = np.array([km.transform([list(map(float, r))])[0] for r in rows])
        b = np.array([rbf.transform([list(map(float, r))])[0] for r in rrows])
        res[nf] = (a, b)
    np.save(os.path.join(D, f"ct_{core}.npy"), np.array([res], dtype=object), allow_pickle=True)
    print(core, "->", arch)


def compare():
    import pickle
    with open(os.path.join(D, "coretype.pkl"), "rb") as f:
        out = pickle.load(f)
    got = {}
    for c in CORES:
        p = os.path.join(D, f"ct_{c}.npy")
        if os.path.exists(p):
            got[c] = np.load(p, allow_pickle=True)[0]
    base = "SkylakeX"
    for c in got:
        if c == base:
            continue
        for nf, (km, rows, rbf, rrows) in out.items():
            a0, b0 = got[base][nf]
            a1, b1 = got[c][nf]
            C = km.cluster_centers_
            # distances: ulps and K2
            mu, mk, ndiff = 0, 0.0, 0
            for i, r in enumerate(rows):
                for k in range(len(C)):
                    t0, t1 = float(a0[i, k]), float(a1[i, k])
                    if t0 != t1:
                        ndiff += 1
                    mu = max(mu, abs(ordered(t0) - ordered(t1)))
                    s2 = float((r * r).sum() + (C[k] * C[k]).sum() + 2 * np.abs(r * C[k]).sum())
                    mk = max(mk, abs(float(Fraction(t0) ** 2 - Fraction(t1) ** 2)) / (EPS * s2))
            # rbf: ulps and K with S = c*(sum|x w| + |b| + 1)
            W, bo = rbf.random_weights_, rbf.random_offset_
            cc = (2.0 / rbf.n_components) ** 0.5
            ru, rk, rdiff = 0, 0.0, 0
            for i, r in enumerate(rrows):
                S = cc * (np.abs(r[:, None] * W).sum(axis=0) + np.abs(bo) + 1)
                d = np.abs(b0[i] - b1[i]) / (EPS * S)
                rk = max(rk, float(d.max()))
                rdiff += int((b0[i] != b1[i]).sum())
                ru = max(ru, max(abs(ordered(float(x)) - ordered(float(y))) for x, y in zip(b0[i], b1[i])))
            print(f"{base} vs {c:<12} nf={nf:>2}: KMeans lanes differ {ndiff}/{a0.size}, max {mu:.3g} ulps, K2 max {mk:.3f}"
                  f" | RBF lanes differ {rdiff}/{b0.size}, max {ru:.3g} ulps, K max {rk:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "run":
        run(sys.argv[2])
    elif cmd == "all":
        prep()
        for c in CORES:
            env = dict(os.environ, OPENBLAS_CORETYPE=c, OPENBLAS_NUM_THREADS="1")
            subprocess.run([sys.executable, os.path.abspath(__file__), "run", c], env=env, check=True)
        compare()
    else:
        compare()
