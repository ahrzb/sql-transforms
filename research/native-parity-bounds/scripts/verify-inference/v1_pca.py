"""V1: is the left-to-right PCA spelling bit-identical to the twin under
OPENBLAS_CORETYPE=Sandybridge on NEW data (other p, other seeds, nonzero
means)?  make -> data/v1.pkl ; then twin_run under envs ; then compare."""
import pickle, sys, math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
V = __import__("pathlib").Path(__file__).resolve().parent
EPS = np.finfo(float).eps

def my_native(est, X, mc=None):
    C = est.components_
    if mc is None:
        mc = (est.mean_.reshape(1, -1) @ C.T)[0]
    sc = np.sqrt(est.explained_variance_); sc[sc < EPS] = EPS
    acc = np.zeros((len(X), C.shape[0]))
    for i in range(X.shape[1]):        # x0*c0 + x1*c1 + ... (0.0 + a == a)
        acc = acc + X[:, i:i+1] * C[None, :, i]
    return (acc - mc) / sc

def pure_python_row(est, x, mc):
    sc = [max(math.sqrt(v), EPS) for v in est.explained_variance_]
    out = []
    for k, c in enumerate(est.components_.tolist()):
        s = x[0] * c[0]
        for a, b in zip(x[1:], c[1:]):
            s = s + a * b
        out.append((s - mc[k]) / sc[k])
    return out

def make():
    rng = np.random.default_rng(777)
    cases = {}
    for name, p, n, std, offs in [("p5_std", 5, 20000, True, 0), ("p12_raw_offset", 12, 20000, False, 1),
                                  ("p33_std", 33, 20000, True, 0), ("p100_std", 100, 12000, True, 0),
                                  ("p100_raw_offset", 100, 12000, False, 1), ("p257_std", 257, 6000, True, 0)]:
        X, y = make_classification(n_samples=n + 8000, n_features=p, n_informative=max(2, p // 3),
                                   n_redundant=0, random_state=int(rng.integers(1e6)))
        if offs:  # raw business-like columns: large means, varied scales, 2 decimals
            X = np.round(X * rng.uniform(1, 500, p) + rng.uniform(-1e4, 1e5, p), 2)
        if std:
            X = (X - X[:n].mean(0)) / X[:n].std(0)
        est = PCA(whiten=True).fit(X[:n])
        cases[name] = dict(est=est, X=X[n:])
    pickle.dump(cases, open(V / "data" / "v1.pkl", "wb"))

def compare():
    from collections import OrderedDict
    cases = pickle.load(open(V / "data" / "v1.pkl", "rb"))
    envs = ["default", "sandybridge", "haswell"]
    Z = {e: np.load(V / "data" / f"v1_{e}.npz") for e in envs}
    for name, c in cases.items():
        est, X = c["est"], c["X"]
        N = my_native(est, X)
        # pure-python spot check of my numpy spelling
        mcd = (est.mean_.reshape(1, -1) @ est.components_.T)[0]
        for r in (0, 1, 2, len(X) - 1):
            assert np.array_equal(np.array(pure_python_row(est, X[r].tolist(), mcd.tolist())), N[r])
        Nsb = my_native(est, X, mc=Z["sandybridge"][name + "__mc"])
        line = f"{name:16s} mean|mu|={np.abs(est.mean_).mean():.2g} mc(default)==mc(SB): {np.array_equal(mcd, Z['sandybridge'][name+'__mc'])} |"
        for e in envs:
            for m in ("row", "batch"):
                T = Z[e][f"{name}__{m}"]
                line += f" {e[:3]}/{m}: N {np.mean(N != T):.3f} N(mcSB) {np.mean(Nsb != T):.3f};"
        print(line)

if __name__ == "__main__":
    {"make": make, "compare": compare}[sys.argv[1]]()
