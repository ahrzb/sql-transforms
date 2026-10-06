"""Q3c: the SplineTransformer(degree=0, extrapolation='constant') sparse/dense
disagreement: which side is 'right', and is it per-row or also batch?"""
import warnings

import numpy as np
from sklearn.preprocessing import SplineTransformer

warnings.filterwarnings("ignore")
X = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
for n_knots in (2, 3, 4):
    for extrap in ("constant", "linear"):
        kw = dict(degree=0, n_knots=n_knots, extrapolation=extrap)
        s = SplineTransformer(sparse_output=True, **kw).fit(X)
        d = SplineTransformer(sparse_output=False, **kw).fit(X)
        probe = np.array([[-0.5], [0.0], [0.5], [1.0], [1.5]])
        print(kw, "t =", d.bsplines_[0].t)
        print("   sparse batch:", s.transform(probe).toarray().tolist())
        print("   dense  batch:", d.transform(probe).tolist())
        print("   sparse rows :", [s.transform([[v]]).toarray()[0].tolist() for v in probe[:, 0]])
