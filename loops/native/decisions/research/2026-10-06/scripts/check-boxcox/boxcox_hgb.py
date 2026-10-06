"""Independent check: does the shipped Box-Cox spelling flip HGB predictions on
grid-valued data, where the twin on the same machine flips none?"""
import math, sys
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import HistGradientBoostingClassifier

def native(x, lam):  # power.py's _box_cox + expm1, glibc via math
    if math.isnan(x) or x <= 0: return math.nan
    lx = math.log(x)
    if abs(lam) < 1e-19: return lx
    w = lam * lx
    if not (w < 709.78):
        b = math.exp(w - math.log(abs(lam)))
        return (b if lam > 0 else -b) - 1 / lam
    u = math.exp(w)
    if u == 1.0: e = w
    elif u - 1.0 == -1.0: e = -1.0
    else: e = (u - 1.0) * (w / math.log(u))
    return e / lam

def draw(rng, n):
    return np.column_stack([
        np.round(rng.lognormal(3, 1, n), 2) + 0.01,      # prices in cents
        rng.poisson(20, n) + 1.0,                         # counts
        rng.integers(1, 1000, n).astype(float),           # scores
        np.round(rng.gamma(2, 50, n), 1) + 0.1,           # one decimal
    ])

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(seed)
Xtr, Xte = draw(rng, 50_000), draw(rng, 200_000)
logit = (np.log(Xtr[:, 0]) - 3) + 0.05 * (Xtr[:, 1] - 20) + (Xtr[:, 2] > 500) - 0.01 * Xtr[:, 3]
ytr = (logit + rng.logistic(size=len(logit)) > 0).astype(int)
pt = PowerTransformer(method="box-cox", standardize=False)
Ftr = pt.fit_transform(Xtr)
hgb = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Ftr, ytr)
T = pt.transform(Xte)                                    # twin, batch
Trow = np.vstack([pt.transform(Xte[i:i + 1]) for i in range(2000)])
N = np.array([[native(v, l) for v, l in zip(r, pt.lambdas_)] for r in Xte])
print("seed", seed, "lambdas", np.round(pt.lambdas_, 3))
print("twin row == batch on 2000 rows:", np.array_equal(Trow, T[:2000]))
print("twin transform == fit_transform on train:", np.array_equal(pt.transform(Xtr), Ftr))
print("entries N != T: %.2f%%" % (100 * np.mean(N != T)))
rt, rn = hgb._raw_predict(T), hgb._raw_predict(N)
print("rows with HGB raw score change:", int(np.sum(np.any(rt != rn, axis=1))), "of", len(T))
print("labels flipped:", int(np.sum(hgb.predict(T) != hgb.predict(N))))
seen = np.isin(Xte, Xtr).all(axis=1)
print("fresh rows whose every value appears in training: %.1f%%" % (100 * seen.mean()))
