"""V2: chi2 native (glibc log/cos/sin via math, per element) vs twin with and
without numpy AVX-512, on new data incl. sample_steps=3 and wide ranges."""
import math, pickle, sys
import numpy as np
from sklearn.kernel_approximation import AdditiveChi2Sampler
V = __import__("pathlib").Path(__file__).resolve().parent

def my_native(est, X, glibc_cosh=False):
    s = float(est.sample_interval_) if getattr(est, "sample_interval_", None) is not None else {1: .8, 2: .5, 3: .4}[est.sample_steps]
    L = est.sample_steps
    ch = [math.cosh(math.pi * j * s) for j in range(L)] if glibc_cosh else [float(np.cosh(np.pi * j * s)) for j in range(L)]
    out = np.zeros((X.shape[0], X.shape[1] * (2 * L - 1)))
    p = X.shape[1]
    for r in range(X.shape[0]):
        for c in range(p):
            x = float(X[r, c])
            if x == 0.0:
                continue
            out[r, c] = math.sqrt(x * s)
            lg = s * math.log(x)
            for j in range(1, L):
                f = math.sqrt((2 * x * s) / ch[j])
                out[r, p * (2 * j - 1) + c] = f * math.cos(j * lg)
                out[r, p * (2 * j) + c] = f * math.sin(j * lg)
    return out

def make():
    rng = np.random.default_rng(4242)
    n = 20000
    cases = {}
    X = np.column_stack([rng.lognormal(0, 2, n), rng.uniform(1e-6, 1e6, n), np.round(rng.lognormal(4, 1.5, n), 2),
                         rng.integers(0, 5000, n).astype(float), np.round(rng.gamma(2, 3, n), 1), rng.random(n) ** 8])
    for steps in (2, 3):
        est = AdditiveChi2Sampler(sample_steps=steps).fit(X)
        cases[f"steps{steps}"] = dict(est=est, X=X)
    pickle.dump(cases, open(V / "data" / "v2.pkl", "wb"))

def compare():
    cases = pickle.load(open(V / "data" / "v2.pkl", "rb"))
    envs = ["default", "noavx512", "sse"]
    Z = {e: np.load(V / "data" / f"v2_{e}.npz") for e in envs}
    for name, c in cases.items():
      for gc in (False, True):
        N = my_native(c["est"], c["X"], gc)
        p = c["X"].shape[1]
        s = f"{name} cosh={'glibc' if gc else 'numpy-default'}:"
        for e in envs:
            for m in ("row", "batch"):
                T = Z[e][f"{name}__{m}"]
                d = N != T
                s += f" {e}/{m} {int(d.sum())}/{d.size} {[int(d[:, k*p:(k+1)*p].sum()) for k in range(N.shape[1]//p)]};"
        print(s)

if __name__ == "__main__":
    {"make": make, "compare": compare}[sys.argv[1]]()
