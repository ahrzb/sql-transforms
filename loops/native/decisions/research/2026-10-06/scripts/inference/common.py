"""Shared: datasets, family configs, the native-like spellings, metrics.

Native-like (N) = the SQL entry's operation order, in IEEE float64 with no
FMA: numpy elementwise ufuncs (`*`, `+`, `-`, `/`, sqrt are correctly
rounded and never fused) in an explicit left-to-right loop over features,
and glibc's log/exp/cos/sin through Python's `math` (DuckDB's ln/exp).
"""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FEATS = ROOT / "feats"
EPS = np.finfo(np.float64).eps
N_TEST = 50_000


# ----------------------------------------------------------------- datasets
def _decimals(col: np.ndarray) -> int:
    for d in range(0, 7):
        s = col * 10.0**d
        if np.all(np.abs(s - np.round(s)) < 1e-6):
            return d
    return 6


def _augment(X: np.ndarray, n: int, rng, integer: bool = False, rel: float = 0.1):
    """Test rows like the data: a resampled row plus jitter of `rel` of the
    column's spread, rounded to the column's recorded precision, clipped to
    its observed range (keeps non-negative columns non-negative)."""
    idx = rng.integers(0, len(X), n)
    base = X[idx]
    sd = X.std(axis=0)
    lo, hi = X.min(axis=0), X.max(axis=0)
    if integer:
        jit = rng.integers(-1, 2, size=base.shape) * (rng.random(base.shape) < 0.15)
        out = base + jit
    else:
        out = base + rng.normal(size=base.shape) * sd * rel
        for j in range(X.shape[1]):
            out[:, j] = np.round(out[:, j], _decimals(X[:, j]))
    return np.clip(out, lo, hi)


def datasets():
    """name -> dict(Xtr, ytr, Xte, yte_reg?, kind) with ~5e4 test rows plus
    the training rows themselves (scoring the training table)."""
    from sklearn import datasets as D

    rng = np.random.default_rng(12345)
    out = {}
    for name, loader, integer in [
        ("breast", D.load_breast_cancer, False),
        ("wine", D.load_wine, False),
        ("digits", D.load_digits, True),
        ("diabetes", D.load_diabetes, False),
    ]:
        b = loader()
        X, y = b.data.astype(np.float64), b.target
        if name == "diabetes":
            yreg = y.astype(float)
            yclf = (y > np.median(y)).astype(int)
        else:
            yclf = y
            yreg = y.astype(float)
        Xa = _augment(X, N_TEST, rng, integer=integer)
        out[name] = dict(Xtr=X, yclf=yclf, yreg=yreg, Xte=np.vstack([Xa, X]))
    from sklearn.datasets import make_classification

    Xs, ys = make_classification(
        n_samples=20_000 + N_TEST, n_features=20, n_informative=8, n_redundant=0,
        n_classes=3, n_clusters_per_class=2, class_sep=0.8, random_state=7,
    )
    out["synth"] = dict(Xtr=Xs[:20_000], yclf=ys[:20_000], yreg=ys[:20_000].astype(float),
                        Xte=np.vstack([Xs[20_000:], Xs[:20_000]]))
    # rank-deficient: 4 redundant columns are exact linear combinations
    Xr, yr = make_classification(
        n_samples=20_000 + N_TEST, n_features=20, n_informative=8, n_redundant=4,
        n_classes=2, class_sep=0.8, random_state=8,
    )
    out["synth_redund"] = dict(Xtr=Xr[:20_000], yclf=yr[:20_000], yreg=yr[:20_000].astype(float),
                               Xte=np.vstack([Xr[20_000:], Xr[:20_000]]))
    # positive heavy-tailed (counts/amounts-like) for Yeo-Johnson / chi2
    Xp = np.round(np.exp(0.7 * Xs + 1.0), 3)
    out["synth_pos"] = dict(Xtr=Xp[:20_000], yclf=ys[:20_000], yreg=ys[:20_000].astype(float),
                            Xte=np.vstack([Xp[20_000:], Xp[:20_000]]))
    return out


# ------------------------------------------------------------- family setup
def standardize(d):
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(d["Xtr"])
    return sc.transform(d["Xtr"]), sc.transform(d["Xte"])


def make_transformer(family: str):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.kernel_approximation import AdditiveChi2Sampler
    from sklearn.preprocessing import PowerTransformer

    if family == "pca_all":
        return PCA(whiten=True)
    if family == "pca_95":
        return PCA(n_components=0.95, whiten=True)
    if family == "kmeans8":
        return KMeans(n_clusters=8, n_init=4, random_state=0)
    if family == "kmeans64":
        return KMeans(n_clusters=64, n_init=2, random_state=0)
    if family == "yj_std":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    if family == "chi2":
        return AdditiveChi2Sampler(sample_steps=2)
    raise KeyError(family)


# family -> (datasets, input is standardized?)
PLAN = {
    "pca_all": (["breast", "wine", "digits", "diabetes", "synth", "synth_redund"], True),
    "pca_95": (["breast", "wine", "digits", "diabetes", "synth", "synth_redund"], True),
    "kmeans8": (["breast", "wine", "digits", "diabetes", "synth"], True),
    "kmeans64": (["breast", "digits", "synth"], True),
    "yj_std": (["breast", "wine", "digits", "diabetes", "synth", "synth_pos"], False),
    "chi2": (["breast", "wine", "digits", "synth_pos"], False),
}


def case_path(family, ds):
    return DATA / f"{family}__{ds}.pkl"


def load_case(family, ds):
    with open(case_path(family, ds), "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------- native-like spellings
def lr_dot(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(rows x n) . (k x n)^T summed left to right, x0*w0 + x1*w1 + ..., as
    SQL `a + b + c` does: one rounded product, one rounded add per term."""
    acc = X[:, 0:1] * W[None, :, 0]
    for i in range(1, X.shape[1]):
        acc = acc + X[:, i : i + 1] * W[None, :, i]
    return acc


def lr_abs_dot(X, W):
    acc = np.abs(X[:, 0:1] * W[None, :, 0])
    for i in range(1, X.shape[1]):
        acc = acc + np.abs(X[:, i : i + 1] * W[None, :, i])
    return acc


def native_pca(est, X):
    mc = (est.mean_.reshape(1, -1) @ est.components_.T)  # a constant, numpy as the twin
    scale = np.sqrt(est.explained_variance_)
    scale[scale < EPS] = EPS
    if not est.whiten:
        scale = np.ones_like(scale)
    xc = lr_dot(X, est.components_)
    out = (xc - mc) / scale
    S = (lr_abs_dot(X, est.components_) + np.abs(mc)) / scale
    return out, S


def native_kmeans(est, X):
    """The twin's formula, sqrt(max((-2 x.c + |x|^2) + |c|^2, 0)), in left to
    right order; |c|^2 a constant (numpy einsum, as the twin)."""
    C = est.cluster_centers_
    YY = np.einsum("ij,ij->i", C, C)[None, :]
    d = lr_dot(X, C)
    xx = X[:, 0] * X[:, 0]
    for i in range(1, X.shape[1]):
        xx = xx + X[:, i] * X[:, i]
    D = (-2.0 * d + xx[:, None]) + YY
    S = 2.0 * lr_abs_dot(X, C) + xx[:, None] + YY  # term scale of D
    return np.sqrt(np.maximum(D, 0.0)), S, D


def native_kmeans_direct(est, X):
    """Alternative spelling: sum (x_i - c_i)^2 left to right, then sqrt."""
    C = est.cluster_centers_
    acc = (X[:, 0:1] - C[None, :, 0]) ** 2
    for i in range(1, X.shape[1]):
        t = X[:, i : i + 1] - C[None, :, i]
        acc = acc + t * t
    return np.sqrt(acc)


def _log1p_goldberg(z: float) -> float:
    u = 1.0 + z
    if u == 1.0:
        return z
    return z * (math.log(u) / (u - 1.0))


def _expm1_kahan(w: float) -> float:
    if w >= 709.78:
        return math.inf
    u = math.exp(w)
    if u == 1.0:
        return w
    if u - 1.0 == -1.0:
        return -1.0
    return (u - 1.0) * (w / math.log(u))


def _yj_one(x: float, lam: float) -> float:
    if math.isnan(x):
        return math.nan
    if x >= 0:
        if abs(lam) < EPS:
            return _log1p_goldberg(x)
        return _expm1_kahan(lam * _log1p_goldberg(x)) / lam
    if abs(lam - 2) > EPS:
        return -_expm1_kahan((2 - lam) * _log1p_goldberg(-x)) / (2 - lam)
    return -_log1p_goldberg(-x)


def native_yj(est, X):
    lams = [float(v) for v in est.lambdas_]
    T = np.empty_like(X)
    St = np.empty_like(X)
    for j, lam in enumerate(lams):
        col = X[:, j]
        for r in range(len(col)):
            x = float(col[r])
            t = _yj_one(x, lam)
            T[r, j] = t
            l1 = _log1p_goldberg(abs(x))
            g = lam if x >= 0 else 2 - lam
            St[r, j] = (1 + abs(g * l1)) * abs(t)
    if est.standardize:
        m, s = est._scaler.mean_, est._scaler.scale_
        out = (T - m) / s
        S = (St + np.abs(m)) / s
    else:
        out, S = T, St
    return out, S


def native_chi2(est, X):
    s = float(est.sample_interval_) if hasattr(est, "sample_interval_") else None
    if s is None:
        s = {1: 0.8, 2: 0.5, 3: 0.4}[est.sample_steps]
    steps = est.sample_steps
    n, p = X.shape
    lanes = [np.zeros_like(X) for _ in range(1 + 2 * (steps - 1))]
    Ss = [np.zeros_like(X) for _ in lanes]
    cosh = [float(np.cosh(np.pi * j * s)) for j in range(steps)]
    for r in range(n):
        for c in range(p):
            x = float(X[r, c])
            if x == 0.0:
                continue
            lanes[0][r, c] = math.sqrt(x * s)
            Ss[0][r, c] = abs(lanes[0][r, c])
            ls = s * math.log(x)
            st = 2 * x * s
            for j in range(1, steps):
                f = math.sqrt(st / cosh[j])
                a = j * ls
                lanes[2 * j - 1][r, c] = f * math.cos(a)
                lanes[2 * j][r, c] = f * math.sin(a)
                sc = f * (1 + abs(a))
                Ss[2 * j - 1][r, c] = sc
                Ss[2 * j][r, c] = sc
    return np.hstack(lanes), np.hstack(Ss)


# ---------------------------------------------------------------- metrics
def ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in representable doubles (0 for equal, NaN==NaN, -0 == 0)."""
    a = np.atleast_1d(np.asarray(a, np.float64))
    b = np.atleast_1d(np.asarray(b, np.float64))
    a = np.where(a == 0, 0.0, a)
    b = np.where(b == 0, 0.0, b)
    ia, ib = a.view(np.int64), b.view(np.int64)
    M = np.int64(0x7FFFFFFFFFFFFFFF)
    ka = np.where(ia < 0, -(ia & M), ia)
    kb = np.where(ib < 0, -(ib & M), ib)
    same = (ka >= 0) == (kb >= 0)
    with np.errstate(over="ignore"):
        d_same = np.abs(ka - kb).astype(np.float64)
    d_opp = np.abs(ka.astype(np.float64) - kb.astype(np.float64))
    d = np.where(same, d_same, d_opp)
    d[np.isnan(a) & np.isnan(b)] = 0
    d[np.isnan(a) ^ np.isnan(b)] = np.inf
    return d
