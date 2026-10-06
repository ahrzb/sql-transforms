"""Q3b: with the SAME fitted pipeline (fitted on the sparse producer), does a
downstream step's transform give the same bits on the sparse row as on the
densified row? This isolates the transform-arithmetic difference a catalog
composition (which reads fitted state and emits dense SQL) would face.
Also: explicit stored zeros in neighbour graphs (densify is lossy there)."""
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer
from sklearn.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    MaxAbsScaler,
    Normalizer,
    OneHotEncoder,
    PolynomialFeatures,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2)
Xs = rng.choice(["a", "b", "c", "d"], size=(300, 2)).astype(object)
Xf = rng.uniform(-2, 3, size=(300, 2))


def bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64)


producers = [
    ("OHE", OneHotEncoder(handle_unknown="ignore"), Xs),
    ("KBins(onehot)", KBinsDiscretizer(encode="onehot", quantile_method="averaged_inverted_cdf"), Xf),
    ("Spline(d3, sparse)", SplineTransformer(sparse_output=True), Xf),
    ("RTE", RandomTreesEmbedding(n_estimators=5, random_state=0), Xf),
]
consumers = [
    ("MaxAbsScaler", MaxAbsScaler()),
    ("StandardScaler(with_mean=False)", StandardScaler(with_mean=False)),
    ("RobustScaler(with_centering=False)", RobustScaler(with_centering=False)),
    ("Normalizer(l2)", Normalizer("l2")),
    ("Normalizer(l1)", Normalizer("l1")),
    ("Normalizer(max)", Normalizer("max")),
    ("PolynomialFeatures(2)", PolynomialFeatures(2)),
    ("Binarizer(0.3)", Binarizer(threshold=0.3)),
    ("VarianceThreshold", VarianceThreshold()),
    ("SimpleImputer", SimpleImputer()),
]
print(f"{'producer -> consumer':55s} rows lanes diff_rows diff_lanes max_ulps")
for pn, prod, X in producers:
    p = clone(prod).fit(X)
    Z = p.transform(X)  # sparse
    for cn, cons in consumers:
        c = clone(cons).fit(Z)  # fitted state as the sparse pipeline has it
        rows = X[:200]
        nd = nl = 0
        lanes = 0
        mx = 0
        for r in rows:
            z = p.transform([list(r)])
            try:
                a = c.transform(z)
                b = c.transform(z.toarray())
            except Exception as e:  # noqa: BLE001
                print(f"  {pn} -> {cn}: {type(e).__name__}: {str(e)[:70]}")
                break
            a = a.toarray() if sp.issparse(a) else np.asarray(a)
            b = b.toarray() if sp.issparse(b) else np.asarray(b)
            lanes += a.size
            d = bits(a) != bits(b)
            if d.any():
                nd += 1
                nl += int(d.sum())
                ia = bits(a).astype(np.int64)
                ib = bits(b).astype(np.int64)
                mx = max(mx, int(np.max(np.abs(ia - ib)[d])))
        else:
            print(f"{pn + ' -> ' + cn:55s} {len(rows):4d} {lanes:6d} {nd:6d} {nl:8d} {mx}")

# Explicit zeros: a neighbour-distance graph stores 0.0 for a neighbour at
# distance 0; densify maps it to the same 0.0 as 'not a neighbour'.
Xk = rng.normal(size=(50, 3))
for name, est in (("KNeighborsTransformer(mode='distance', n_neighbors=3)", KNeighborsTransformer(n_neighbors=3)),
                  ("RadiusNeighborsTransformer(radius=1.0)", RadiusNeighborsTransformer(radius=1.0))):
    est.fit(Xk)
    g = est.transform(Xk[:1])  # row equal to a training row
    print(f"\n{name}: shape {g.shape}, nnz {g.nnz}, stored data {np.round(g.data, 4).tolist()}")
    print("  stored zeros:", int(np.sum(g.data == 0)), "-> dense row has",
          int(np.sum(g.toarray()[0] == 0)), "zeros of", g.shape[1], "(structure lost)")
