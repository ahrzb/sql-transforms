import warnings; warnings.simplefilter("ignore")
import numpy as np, scipy.sparse as sp, sklearn, itertools
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
rng = np.random.default_rng(123)
def bits(a): return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)
tot = dict(cfg=0, rows=0, diff=0, stored_zero=0, negzero_stored=0)
# OHE configs
Xn = rng.choice([-0.0, 0.0, 1.5, 2.0, np.nan, 3.25, -7.0], size=(400, 3)).astype(float)
Xs = rng.choice(["a", "b", "c", None, "d", "e"], size=(400, 2)).astype(object)
Xs[Xs == None] = np.nan  # noqa
probeN = np.vstack([Xn[:150], rng.choice([9.0, -0.0, 0.0, np.nan, 1.5], size=(50, 3))])
probeS = np.vstack([Xs[:150], rng.choice(["zz", "a", "e"], size=(50, 2)).astype(object)])
cfgs = []
for hu in ("ignore", "infrequent_if_exist", "warn"):
    for drop in (None, "first", "if_binary"):
        for mf in (None, 30, 0.2):
            for mc in (None, 3):
                for dt in (np.float64, np.float32, np.int64):
                    cfgs.append(dict(handle_unknown=hu, drop=drop, min_frequency=mf, max_categories=mc, dtype=dt))
for iface in ("spmatrix", "sparray"):
  with sklearn.config_context(sparse_interface=iface):
    for c in cfgs:
        for X, P in ((Xn, probeN), (Xs, probeS)):
            try:
                a = OneHotEncoder(sparse_output=True, **c).fit(X)
                b = OneHotEncoder(sparse_output=False, **c).fit(X)
            except Exception:
                continue
            tot["cfg"] += 1
            for r in P:
                try:
                    sa = a.transform([r]); db = b.transform([r])
                except Exception as e:
                    # both must raise
                    try: b.transform([r]); print("ASYM raise", c)
                    except Exception: pass
                    continue
                assert sp.issparse(sa)
                da = sa.toarray()
                tot["rows"] += 1
                if (sa.data == 0).any(): tot["stored_zero"] += 1
                if np.signbit(sa.data[sa.data == 0]).any(): tot["negzero_stored"] += 1
                if da.dtype != db.dtype or not np.array_equal(bits(da), bits(db)):
                    tot["diff"] += 1
print("OHE", tot)
tot = dict(cfg=0, rows=0, diff=0)
X = np.concatenate([rng.normal(size=(300, 2)) * 1e3, rng.choice([-0.0, 0.0, 5e-324, -5e-324, 1e300, -1e300], size=(300, 1))], axis=1)
P = np.vstack([X, [[1e308, -1e308, -0.0], [-0.0, 0.0, 5e-324], [np.inf, -np.inf, 0.0]]])
for iface in ("spmatrix", "sparray"):
  with sklearn.config_context(sparse_interface=iface):
    for strat in ("uniform", "quantile", "kmeans"):
        for nb in (2, 4, 7, 11):
            for dt in (None, np.float32):
                kw = dict(n_bins=nb, strategy=strat, dtype=dt)
                if strat == "quantile": kw["quantile_method"] = "averaged_inverted_cdf"
                a = KBinsDiscretizer(encode="onehot", **kw).fit(X)
                b = KBinsDiscretizer(encode="onehot-dense", **kw).fit(X)
                tot["cfg"] += 1
                for r in P:
                    try: sa = a.transform([r])
                    except Exception:
                        try: b.transform([r]); print("ASYM")
                        except Exception: pass
                        continue
                    db = b.transform([r]); da = sa.toarray(); tot["rows"] += 1
                    if da.dtype != db.dtype or not np.array_equal(bits(da), bits(db)): tot["diff"] += 1
print("KBins", tot)
