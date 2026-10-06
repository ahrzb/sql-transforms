"""V3 shared: my own data generators and my own native spellings."""
import math
import numpy as np
EPS = np.finfo(float).eps


def gen_txn(n, rng):
    """business-like positive columns on recorded-precision grids"""
    amount = np.round(rng.lognormal(3.5, 1.2, n), 2)
    qty = rng.negative_binomial(3, 0.01, n).astype(float)
    tenure = rng.integers(0, 3650, n).astype(float)
    rate = np.round(rng.beta(2, 5, n) * 100, 1)
    score = np.clip(np.round(rng.normal(600, 80, n)), 300, 850)
    ratio = np.round(rng.lognormal(0, 0.5, n), 3)
    X = np.column_stack([amount, qty, tenure, rate, score, ratio])
    z = (0.9 * np.log1p(amount) - 3.2 + 0.003 * (qty - 300) - 0.0003 * (tenure - 1800)
         + 0.02 * (rate - 28) + 0.01 * (score - 600) + 0.8 * np.log(ratio) + rng.logistic(0, 1, n))
    return X, (z > 0).astype(int)


def gen_cont(n, rng):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=15, n_informative=6, n_redundant=0,
                               n_classes=2, class_sep=0.7, flip_y=0.05,
                               random_state=int(rng.integers(1 << 30)))
    return X, y


# ------------------------------------------------------------- native spellings
def l2r_dot(X, W):
    acc = X[:, 0:1] * W[None, :, 0]
    for i in range(1, X.shape[1]):
        acc = acc + X[:, i:i + 1] * W[None, :, i]
    return acc


def nat_pca(est, X):
    C = est.components_
    mc = (est.mean_.reshape(1, -1) @ C.T)[0]
    sc = np.sqrt(est.explained_variance_) if est.whiten else np.ones(C.shape[0])
    sc[sc < EPS] = EPS
    return (l2r_dot(X, C) - mc) / sc


def nat_kmeans(est, X):
    C = est.cluster_centers_
    cc = np.einsum("ij,ij->i", C, C)
    xx = X[:, 0] * X[:, 0]
    for i in range(1, X.shape[1]):
        xx = xx + X[:, i] * X[:, i]
    D = (-2.0 * l2r_dot(X, C) + xx[:, None]) + cc[None, :]
    return np.sqrt(np.maximum(D, 0.0))


def _log1p(z):
    u = 1.0 + z
    return z if u == 1.0 else z * (math.log(u) / (u - 1.0))


def _expm1(w):
    u = math.exp(w)
    if u == 1.0:
        return w
    if u - 1.0 == -1.0:
        return -1.0
    return (u - 1.0) * (w / math.log(u))


def nat_yj(est, X):
    out = np.empty_like(X)
    for j, lam in enumerate(est.lambdas_.tolist()):
        col = X[:, j].tolist()
        res = []
        for x in col:
            if x >= 0:
                t = _log1p(x) if abs(lam) < EPS else _expm1(lam * _log1p(x)) / lam
            else:
                t = -_expm1((2 - lam) * _log1p(-x)) / (2 - lam) if abs(lam - 2) > EPS else -_log1p(-x)
            res.append(t)
        out[:, j] = res
    if est.standardize:
        out = (out - est._scaler.mean_) / est._scaler.scale_
    return out


def nat_chi2(est, X):
    s = float(est.sample_interval_) if getattr(est, "sample_interval_", None) is not None else {1: .8, 2: .5, 3: .4}[est.sample_steps]
    L = est.sample_steps
    ch = [float(np.cosh(np.pi * j * s)) for j in range(L)]
    n, p = X.shape
    out = np.zeros((n, p * (2 * L - 1)))
    for c in range(p):
        for r, x in enumerate(X[:, c].tolist()):
            if x == 0.0:
                continue
            out[r, c] = math.sqrt(x * s)
            lg = s * math.log(x)
            for j in range(1, L):
                f = math.sqrt((2 * x * s) / ch[j])
                out[r, p * (2 * j - 1) + c] = f * math.cos(j * lg)
                out[r, p * (2 * j) + c] = f * math.sin(j * lg)
    return out


def native(fam, est, X):
    return {"yj": nat_yj, "chi2": nat_chi2, "kmeans": nat_kmeans, "pca": nat_pca}[fam.split("_")[0].replace("pcaraw", "pca")](est, X)
