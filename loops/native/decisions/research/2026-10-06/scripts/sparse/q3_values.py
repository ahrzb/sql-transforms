"""Q3: is a densified sparse row bit-identical to the dense configuration's
row? Compared per row, as the step calls transform([row]), on the raw
float64 bit patterns (so -0.0 != 0.0 and NaN payloads count)."""
import itertools
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.impute import MissingIndicator
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    MaxAbsScaler,
    Normalizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1)


def bits(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return a.view(np.uint64)


def step_row(est, row):
    """What a densifying step would hand on: transform([row]), densified,
    first row, as float()s."""
    out = est.transform([row])
    if sp.issparse(out):
        out = out.toarray()
    return np.array([float(v) for v in np.asarray(out)[0]])


def compare(label, sparse_est, dense_est, Xfit, Xrows, yfit=None):
    s = sparse_est.fit(Xfit, yfit) if yfit is not None else sparse_est.fit(Xfit)
    d = dense_est.fit(Xfit, yfit) if yfit is not None else dense_est.fit(Xfit)
    n = lanes = diff_rows = negz = stored_zero = 0
    err_mismatch = 0
    first = None
    for r in Xrows:
        r = list(r)
        try:
            a = step_row(s, r)
            ea = None
        except Exception as e:  # noqa: BLE001
            ea = type(e).__name__
        try:
            b = step_row(d, r)
            eb = None
        except Exception as e:  # noqa: BLE001
            eb = type(e).__name__
        if ea or eb:
            err_mismatch += (ea is None) != (eb is None)
            continue
        n += 1
        lanes += a.size
        raw = s.transform([r])
        if sp.issparse(raw):
            data = raw.tocsr().data
            stored_zero += int(np.sum(data == 0))
            negz += int(np.sum(np.signbit(data) & (data == 0)))
        if a.shape != b.shape or np.any(bits(a) != bits(b)):
            diff_rows += 1
            if first is None:
                idx = np.flatnonzero(bits(a) != bits(b)) if a.shape == b.shape else []
                first = (r, [(int(i), a[i], b[i]) for i in idx[:3]])
    print(f"{label:62s} rows {n:6d} lanes {lanes:8d} differing_rows {diff_rows:5d}"
          f" stored_zeros {stored_zero:5d} (neg {negz}) raise_mismatch {err_mismatch}")
    if first:
        print("     first diff:", first)
    return diff_rows


total = 0
# ---------- OneHotEncoder ----------
Xs = rng.choice(["a", "b", "c", "d", "e", "f"], p=[.4, .3, .15, .1, .04, .01], size=(400, 2)).astype(object)
Xs[::37, 1] = None
Xn = rng.choice([0.0, 1.0, 2.0, -0.0, 3.5, np.nan], size=(400, 2))
probe_s = np.vstack([Xs[:200], np.array([["zz", "a"], ["a", "zz"], ["f", None]], dtype=object)])
probe_n = np.vstack([Xn[:200], [[9.0, 0.0], [-0.0, 7.0], [np.nan, np.nan], [0.0, -0.0]]])
ohe_cfgs = [
    {},
    {"handle_unknown": "ignore"},
    {"handle_unknown": "infrequent_if_exist", "min_frequency": 20},
    {"handle_unknown": "infrequent_if_exist", "max_categories": 3},
    {"handle_unknown": "warn", "min_frequency": 20},
    {"drop": "first", "handle_unknown": "ignore"},
    {"drop": "if_binary"},
    {"drop": "first", "handle_unknown": "infrequent_if_exist", "min_frequency": 20},
    {"dtype": np.float32, "handle_unknown": "ignore"},
    {"dtype": np.int64, "handle_unknown": "ignore"},
]
for cfg in ohe_cfgs:
    for X, P, kind in ((Xs, probe_s, "str"), (Xn, probe_n, "num")):
        total += compare(f"OHE {kind} {cfg}", OneHotEncoder(**cfg), OneHotEncoder(sparse_output=False, **cfg), X, P)

# ---------- KBinsDiscretizer ----------
Xk = rng.normal(size=(300, 3))
Xk[:, 2] = np.round(Xk[:, 2])  # ties, repeated edges
edgey = np.vstack([Xk[:150], [[0.0, -0.0, 0.0], [-0.0, 0.0, -0.0], [1e300, -1e300, 5e-324], [-5e-324, 1e-320, 0.5]]])
for strat in ("quantile", "uniform", "kmeans"):
    for nb in (2, 5, 9):
        kw = dict(n_bins=nb, strategy=strat)
        if strat == "quantile":
            kw["quantile_method"] = "averaged_inverted_cdf"
        if strat == "kmeans":
            kw["subsample"] = None
        total += compare(f"KBins {kw}", KBinsDiscretizer(encode="onehot", **kw),
                         KBinsDiscretizer(encode="onehot-dense", **kw), Xk, edgey)
# ---------- SplineTransformer ----------
Xsp = rng.uniform(-2, 3, size=(200, 2))
knots_edge = []
for _ in range(1):
    pass
for degree, n_knots, extrap, bias, kn in itertools.product(
        (0, 1, 2, 3, 5), (2, 3, 6), ("constant", "linear", "continue", "periodic", "error"),
        (True, False), ("uniform", "quantile")):
    if extrap == "periodic" and degree == 0:
        continue
    kw = dict(degree=degree, n_knots=n_knots, extrapolation=extrap, include_bias=bias, knots=kn)
    try:
        dense = SplineTransformer(sparse_output=False, **kw).fit(Xsp)
    except Exception:
        continue
    t = dense.bsplines_[0].t
    rows = np.vstack([
        rng.uniform(-4, 5, size=(120, 2)),
        np.column_stack([t, t[::-1]]),                       # exactly on knots
        np.column_stack([np.nextafter(t, -np.inf), np.nextafter(t, np.inf)]),
        [[0.0, -0.0], [-0.0, 0.0], [np.nan, 1.0], [1.0, np.nan], [1e12, -1e12]],
    ])
    total += compare(f"Spline {kw}", SplineTransformer(sparse_output=True, **kw),
                     SplineTransformer(sparse_output=False, **kw), Xsp, rows)

# ---------- MissingIndicator ----------
Xm = rng.normal(size=(100, 3))
Xm[rng.random((100, 3)) < 0.2] = np.nan
for feats in ("missing-only", "all"):
    total += compare(f"MissingIndicator features={feats}", MissingIndicator(sparse=True, features=feats),
                     MissingIndicator(sparse=False, features=feats), Xm, Xm[:60])
# ---------- RandomTreesEmbedding ----------
Xr = rng.normal(size=(200, 3))
total += compare("RandomTreesEmbedding", RandomTreesEmbedding(n_estimators=10, random_state=0),
                 RandomTreesEmbedding(n_estimators=10, random_state=0, sparse_output=False), Xr, Xr[:80])
# ---------- ColumnTransformer: sparse_threshold path vs dense path ----------
Xc = np.column_stack([rng.choice([0.0, 1.0, 2.0], size=200), rng.normal(size=200)])
Xc[:5, 1] = 0.0
probe_c = np.vstack([Xc[:60], [[1.0, -0.0], [2.0, 0.0], [0.0, -0.0]]])
neg = FunctionTransformer(np.negative)
for part2_name, part2 in (("passthrough-ish identity FT", FunctionTransformer()),
                          ("FT(np.negative)", neg),
                          ("StandardScaler(with_mean=False)", StandardScaler(with_mean=False))):
    mk = lambda thr: ColumnTransformer([("o", OneHotEncoder(), [0]), ("p", clone(part2), [1])],
                                       sparse_threshold=thr)
    total += compare(f"ColumnTransformer OHE+{part2_name} thr=1 vs 0", mk(1.0), mk(0.0), Xc, probe_c)
# FeatureUnion: sparse (OHE sparse) vs dense (OHE dense)
for part2_name, part2 in (("FT(np.negative)", neg), ("StandardScaler(with_mean=False)", StandardScaler(with_mean=False))):
    total += compare(f"FeatureUnion OHE+{part2_name}",
                     FeatureUnion([("o", OneHotEncoder()), ("p", clone(part2))]),
                     FeatureUnion([("o", OneHotEncoder(sparse_output=False)), ("p", clone(part2))]),
                     Xc, probe_c)
# ---------- Pipelines with a sparse intermediate ----------
for down_name, down in (("MaxAbsScaler", MaxAbsScaler()),
                        ("StandardScaler(with_mean=False)", StandardScaler(with_mean=False)),
                        ("Normalizer(l2)", Normalizer()), ("Normalizer(l1)", Normalizer("l1")),
                        ("Normalizer(max)", Normalizer("max")),
                        ("PolynomialFeatures(2)", PolynomialFeatures(2))):
    total += compare(f"Pipeline OHE -> {down_name}",
                     make_pipeline(OneHotEncoder(handle_unknown="ignore"), clone(down)),
                     make_pipeline(OneHotEncoder(handle_unknown="ignore", sparse_output=False), clone(down)),
                     Xs, probe_s)
    total += compare(f"Pipeline Spline(d3,k5) -> {down_name}",
                     make_pipeline(SplineTransformer(sparse_output=True), clone(down)),
                     make_pipeline(SplineTransformer(sparse_output=False), clone(down)),
                     Xsp, rng.uniform(-2, 3, size=(400, 2)))
print("TOTAL differing rows:", total)
