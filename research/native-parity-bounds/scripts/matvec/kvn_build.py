"""K vs n: build datasets and fitted models once (pickled), for n in NS.

datasets (k = 16 components, R rows each):
  gauss   : C = orthonormal rows (QR of a Gaussian), mean 0, x ~ N(0, 1)
  offset  : C as gauss, mean mu_i ~ U(-100, 100), x = mu + N(0, 1)    (x.c ~ m.c: cancels)
  pca     : PCA(whiten=True, n_components=16) fitted on mu + Z W^T + noise with
            heterogeneous feature scales (log-uniform 1e-2..1e4) and means; held-out
            rows: half from the distribution, half within 1e-6 * scale of the mean
"""

import pickle
import sys

import numpy as np
from sklearn.decomposition import PCA

NS = [4, 32, 128, 512, 2048]
K = 16
R = 500
rng = np.random.default_rng(2026)
out = {}
for n in NS:
    k = min(K, n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    C = np.ascontiguousarray(Q.T)
    out[("gauss", n)] = dict(C=C, mean=np.zeros(n), X=rng.standard_normal((R, n)))
    mu = rng.uniform(-100, 100, n)
    out[("offset", n)] = dict(C=C, mean=mu, X=mu + rng.standard_normal((R, n)))
    # PCA-like
    scale = np.exp(rng.uniform(np.log(1e-2), np.log(1e4), n))
    mean = rng.uniform(-3, 3, n) * scale * rng.choice([1.0, 100.0], n, p=[0.7, 0.3])
    r = 5
    W = rng.standard_normal((n, r))
    nfit = max(4 * n, 200)
    Z = rng.standard_normal((nfit + R, r)) * np.array([5, 3, 2, 1, 0.5])
    Xall = mean + (Z @ W.T + 0.3 * rng.standard_normal((nfit + R, n))) * scale
    est = PCA(n_components=k, whiten=True, svd_solver="full").fit(Xall[:nfit])
    Xt = Xall[nfit:].copy()
    near = rng.random(R) < 0.5
    Xt[near] = est.mean_ + 1e-6 * scale * rng.standard_normal((near.sum(), n))
    out[("pca", n)] = dict(est=est, C=est.components_, mean=est.mean_, X=Xt)
    print(n, "pca components_ C-contiguous:", est.components_.flags.c_contiguous)
with open(sys.argv[1], "wb") as f:
    pickle.dump(out, f)
