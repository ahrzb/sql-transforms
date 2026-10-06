import numpy as np, scipy.sparse as sp
from sklearn.impute import MissingIndicator
rng = np.random.default_rng(2)
X = rng.normal(size=(100, 4)); X[rng.random((100, 4)) < 0.3] = np.nan; X[:, 3] = 1.0
P = rng.normal(size=(60, 4)); P[rng.random((60, 4)) < 0.4] = np.nan
for feats in ("missing-only", "all"):
    for mv in (np.nan,):
        a = MissingIndicator(sparse=True, features=feats, error_on_new=False).fit(X)
        b = MissingIndicator(sparse=False, features=feats, error_on_new=False).fit(X)
        d = 0
        for r in P:
            sa = a.transform([r]); assert sp.issparse(sa)
            A = sa.toarray(); B = b.transform([r])
            d += int(A.dtype != B.dtype or not np.array_equal(A, B))
        print(feats, "rows differing", d, "/", len(P), "dtype", A.dtype, type(sa).__name__)
