"""Generate fitted PowerTransformer fixtures (catalog generator) and pickle them.
usage: vgen.py SEEDS VARIANT OUT.pkl"""
import sys, typing, pickle, warnings
_e = typing._eval_type
typing._eval_type = lambda *a, prefer_fwd_module=None, **k: _e(*a, **k)
import numpy as np
from scipy import stats
from scipy.special import boxcox
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sql_transform.native import catalog_test as ct
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
E = 2.0 ** -52
YJ_PIN = [0.0, E, -E, E / 2, 2.0, 2 + 2 * E, 2 - E, 2 - 2 * E, 1.0, 1e-3, -1e-3,
          2.001, 1.999, 7.0, -6.0, 0.25, 1e-12, 2 - 1e-12]
BC_PIN = ct.BOX_COX_LAMBDAS + [0.5, -0.5, 1e-10]

def pinned(method, standardize):
    lams = YJ_PIN if method == "yeo-johnson" else BC_PIN
    est = PowerTransformer(method, standardize=standardize)
    def fit(X, y=None):
        del est.fit
        PowerTransformer.fit(est, X, y)
        n = len(est.lambdas_)
        est.lambdas_ = np.array([lams[j % len(lams)] for j in range(n)], dtype=float)
        if standardize:
            X = np.asarray(X, dtype=float)
            f = stats.yeojohnson if method == "yeo-johnson" else boxcox
            Xt = np.column_stack([f(X[:, j].copy(), est.lambdas_[j]) for j in range(n)])
            est._scaler = StandardScaler(copy=False).set_output(transform="default").fit(Xt)
        return est
    est.fit = fit
    return est

CONFIGS = {
    "yj0": (lambda: PowerTransformer(standardize=False), False),
    "yj1": (lambda: PowerTransformer(), False),
    "yjp0": (lambda: pinned("yeo-johnson", False), False),
    "yjp1": (lambda: pinned("yeo-johnson", True), False),
    "bc1": (lambda: PowerTransformer("box-cox"), True),
    "bc0": (lambda: PowerTransformer("box-cox", standardize=False), True),
    "bcp1": (lambda: pinned("box-cox", True), True),
}
N, V, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
data = []
for name, (make, pos) in CONFIGS.items():
    m = ct.positive(make) if pos else make
    for seed in range(N):
        try:
            step = ct._step(m, seed, V)
        except AssertionError:
            continue
        rows = ct._rows(step, seed, pos).to_pylist()
        insts = {}
        for k, est in step.instances.items():
            # strip the monkeypatched fit closure for pickling
            est.__dict__.pop("fit", None)
            insts[k] = est
        data.append(dict(cfg=name, seed=seed, names=step.takes.names, inst=insts, rows=rows))
pickle.dump(data, open(OUT, "wb"))
print(len(data), "steps")
