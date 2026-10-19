import warnings; warnings.simplefilter("ignore")
import numpy as np, scipy.sparse as sp
from sklearn.preprocessing import (SplineTransformer, OneHotEncoder, KBinsDiscretizer, StandardScaler, MaxAbsScaler,
    RobustScaler, Normalizer, Binarizer, PolynomialFeatures)
from sklearn.ensemble import RandomTreesEmbedding
rng = np.random.default_rng(11)
X = rng.uniform(-2, 3, size=(500, 3))
def ulps(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ia = a.view(np.int64); ib = b.view(np.int64)
    ia = np.where(ia < 0, np.int64(-2**63) - ia, ia); ib = np.where(ib < 0, np.int64(-2**63) - ib, ib)
    return np.abs(ia - ib)
producers = {"spline3": SplineTransformer(degree=3, n_knots=6, sparse_output=True),
             "ohe": OneHotEncoder(handle_unknown="ignore"),
             "kbins": KBinsDiscretizer(n_bins=5, quantile_method="averaged_inverted_cdf"),
             "rte": RandomTreesEmbedding(n_estimators=5, random_state=0)}
consumers = {"std(wm=F)": lambda: StandardScaler(with_mean=False), "maxabs": MaxAbsScaler,
             "robust(wc=F)": lambda: RobustScaler(with_centering=False),
             "norm-l1": lambda: Normalizer("l1"), "norm-l2": lambda: Normalizer("l2"), "norm-max": lambda: Normalizer("max"),
             "poly2": lambda: PolynomialFeatures(2)}
for pn, p in producers.items():
    Xp = np.round(X, 1) if pn == "ohe" else X
    p.fit(Xp); S = p.transform(Xp); D = S.toarray()
    for cn, mk in consumers.items():
        c = mk().fit(S)  # fitted on the sparse intermediate as the pipeline would
        a = c.transform(S); a = a.toarray() if sp.issparse(a) else a
        b = c.transform(D); b = b.toarray() if sp.issparse(b) else b
        u = ulps(a, b)
        print(f"{pn:8s} -> {cn:13s} max ulp {u.max()}  lanes differing {(u>0).sum()}/{u.size}")
