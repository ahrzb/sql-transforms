"""E1b: the twin row by row (as PythonTransform calls it) against the same
estimator on the batch of those rows (gemm instead of gemv)."""
import pickle, numpy as np, warnings
from collections import defaultdict
warnings.simplefilter("ignore")
EPS = 2.0**-52
cases = pickle.load(open("pca_cases.pkl", "rb"))
groups = defaultdict(list)
for seed, est, x in cases:
    groups[id(est)].append((est, x))
def ordered(a):
    i = np.asarray(a, np.float64).view(np.int64)
    return np.where(i >= 0, i, -(i & 0x7FFF_FFFF_FFFF_FFFF)).astype(object)
tot = diff = 0; mu = 0; mk = 0.0
for g in groups.values():
    est = g[0][0]; X = np.array([x for _, x in g])
    if len(X) < 2: continue
    B = est.transform(X); R1 = np.vstack([est.transform([x])[0] for x in X])
    fin = np.isfinite(B) & np.isfinite(R1)
    C = est.components_; sc = np.sqrt(est.explained_variance_); sc[sc < EPS] = EPS
    M = (est.mean_.reshape(1, -1) @ C.T)[0]
    S = (np.abs(X[:, None, :] * C[None, :, :]).sum(axis=2) + np.abs(M)) / sc
    tot += int(fin.sum()); diff += int((B[fin] != R1[fin]).sum())
    if fin.any():
        mu = max(mu, int(np.abs(ordered(B[fin]) - ordered(R1[fin])).max()))
        mk = max(mk, float((np.abs(B - R1)[fin] / (EPS * S[fin])).max()))
print(f"PCA(whiten) batch vs row-by-row: lanes {tot}, differ {diff}, max ulps {mu}, max K {mk:.3f}")
# batch against native (left to right, M as the twin computes it)
tot = diff = 0
for g in groups.values():
    est = g[0][0]; X = np.array([x for _, x in g])
    if len(X) < 2: continue
    B = est.transform(X); C = est.components_; sc = np.sqrt(est.explained_variance_); sc[sc < EPS] = EPS
    M = (est.mean_.reshape(1, -1) @ C.T)[0]
    P = X[:, None, :] * C[None, :, :]; N = (np.add.accumulate(P, axis=2)[:, :, -1] - M) / sc
    fin = np.isfinite(B) & np.isfinite(N)
    tot += int(fin.sum()); diff += int((B[fin] != N[fin]).sum())
print(f"PCA(whiten) batch twin vs native left-to-right: lanes {tot}, differ {diff}")
