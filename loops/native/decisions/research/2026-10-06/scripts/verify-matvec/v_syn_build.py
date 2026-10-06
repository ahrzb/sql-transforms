"""Own synthetic data, different from the report's: for each n,
  toep : PCA(whiten, 16) on x = 1000 + AR(1) Toeplitz-correlated N(0,1) (rho .9); all features
         share a large common offset (classic cancellation in x.c - m.c)
  hetero: PCA(whiten, 16) on factor data with per-feature scales 10^U(-3,5), means 10^U(0,6)*sign
  zmean : orthonormal C, mean 0, x ~ N(0,1) (control)
Held-out rows: 60% from the distribution, 40% within 1e-8*sd of the mean."""
import pickle, sys
import numpy as np
from sklearn.decomposition import PCA
rng = np.random.default_rng(777); R = 1000; out = {}
for n in (4, 16, 32, 64, 128, 512, 2048):
    k = min(16, n)
    # toep
    L = np.linalg.cholesky(0.9 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n))) + 1e-9*np.eye(n))
    nf = max(3 * n, 300)
    Xa = 1000 + rng.standard_normal((nf + R, n)) @ L.T
    est = PCA(n_components=k, whiten=True, svd_solver="full").fit(Xa[:nf])
    Xt = Xa[nf:].copy(); near = rng.random(R) < 0.4
    Xt[near] = est.mean_ + 1e-8 * rng.standard_normal((near.sum(), n))
    out[("toep", n)] = dict(est=est, X=Xt)
    # hetero
    sc = 10 ** rng.uniform(-3, 5, n); mu = 10 ** rng.uniform(0, 6, n) * rng.choice([-1, 1], n)
    r = 6; W = rng.standard_normal((n, r))
    Xa = mu + (rng.standard_normal((nf + R, r)) * np.geomspace(4, 0.3, r) @ W.T + 0.2 * rng.standard_normal((nf + R, n))) * sc
    est = PCA(n_components=k, whiten=True, svd_solver="full").fit(Xa[:nf])
    Xt = Xa[nf:].copy(); near = rng.random(R) < 0.4
    Xt[near] = est.mean_ + 1e-8 * sc * rng.standard_normal((near.sum(), n))
    out[("hetero", n)] = dict(est=est, X=Xt)
    # zero-mean control: a PCA object with mean_ = 0 and unit variance
    Q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    est = PCA(n_components=k, whiten=True).fit(rng.standard_normal((max(2*n, 50), n)))
    est.components_ = np.ascontiguousarray(Q.T); est.mean_ = np.zeros(n); est.explained_variance_ = np.ones(k)
    out[("zmean", n)] = dict(est=est, X=rng.standard_normal((R, n)))
pickle.dump(out, open(sys.argv[1], "wb"))
print({k: v["est"].components_.flags.c_contiguous for k, v in out.items() if k[1] == 32})
