"""K = max |native - twin| / (eps * S) over the catalog's fixture generator
(sql_transform.native.catalog_test._step/_rows), seeds 0..N-1, for each
PowerTransformer configuration. The native side is the entry's spelling in
the entry's operation order on glibc's log/exp (bit-identical to DuckDB's,
checked by spellings.py); the twin is `est.transform([row])`, as
PythonTransform calls it.

usage: python kmeasure.py SEEDS OUT.json
"""

import ctypes
import json
import math
import sys
import typing
import warnings

# Python 3.14.0rc2 lacks typing._eval_type(prefer_fwd_module=...), which
# the venv's pydantic passes; drop it so sql_transform imports.
_eval = typing._eval_type
typing._eval_type = lambda *a, prefer_fwd_module=None, **k: _eval(*a, **k)

import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer, StandardScaler

from sql_transform.native import catalog_test as ct

SEEDS = int(sys.argv[1])
OUT = sys.argv[2]
EPS = 2.0**-52
libm = ctypes.CDLL("libm.so.6")
for f in ("log", "exp"):
    getattr(libm, f).restype = ctypes.c_double
    getattr(libm, f).argtypes = [ctypes.c_double]
ln, ex = libm.log, libm.exp
warnings.filterwarnings("ignore")
np.seterr(all="ignore")


def log1p_g(z):  # Goldberg: z * (ln(u) / (u - 1)), u = 1 + z
    u = 1.0 + z
    if u == 1.0:
        return z
    return z * (ln(u) / (u - 1.0))


def expm1_k(w):  # Kahan: (u - 1) * (w / ln(u)), u = exp(w); power.py's arms
    u = ex(w)
    if math.isinf(u):
        return math.inf  # an overflow arm the YJ entry would need
    if u == 1.0:
        return w
    if u - 1.0 == -1.0:
        return -1.0
    return (u - 1.0) * (w / ln(u))


def yj(x, lam):
    """(t, w): scipy's _yeojohnson_transform branches, eps = 2**-52."""
    if math.isnan(x):
        return math.nan, 0.0
    if x >= 0:
        if abs(lam) < EPS:
            return log1p_g(x), 0.0
        w = lam * log1p_g(x)
        return expm1_k(w) / lam, w
    if abs(lam - 2) > EPS:
        d = 2.0 - lam
        w = d * log1p_g(-x)
        return -expm1_k(w) / d, w
    return -log1p_g(-x), 0.0


def bc(x, lam):
    """(t, w): xsf boxcox branches (power.py's _box_cox)."""
    if math.isnan(x) or x <= 0:
        return math.nan, 0.0
    lx = ln(x)
    if abs(lam) < 1e-19:
        return lx, 0.0
    w = lam * lx
    if w < 709.78:
        return expm1_k(w) / lam, 0.0  # log shared: S_t = |t|
    big = ex(w - math.log(abs(lam)))
    return (big if lam > 0 else -big) - 1 / lam, 0.0


E = EPS
YJ_LAMBDAS = [0.0, E / 2, E, -E, 2.0, 2 - E, 2 + 2 * E, 2 - 2 * E, 1.0, -2.0,
              3.0, 4.5, -4.0, 1e-8, 0.5]


def pinned(method, standardize):
    lams = YJ_LAMBDAS if method == "yeo-johnson" else ct.BOX_COX_LAMBDAS
    est = PowerTransformer(method, standardize=standardize)

    def fit(X, y=None):
        del est.fit
        PowerTransformer.fit(est, X, y)
        n = len(est.lambdas_)
        est.lambdas_ = np.array([lams[j % len(lams)] for j in range(n)])
        if standardize:  # the scaler refit on the pinned transform
            f = stats.yeojohnson if method == "yeo-johnson" else None
            from scipy.special import boxcox
            f = f or boxcox
            Xt = np.column_stack([f(X[:, j].astype(float), est.lambdas_[j])
                                  for j in range(n)])
            est._scaler = StandardScaler(copy=False).set_output(
                transform="default").fit(Xt)
        return est

    est.fit = fit
    return est


CONFIGS = {
    "yj_std0": (lambda: PowerTransformer(standardize=False), False),
    "yj_pinned_std0": (lambda: pinned("yeo-johnson", False), False),
    "yj_std1": (lambda: PowerTransformer(), False),
    "yj_pinned_std1": (lambda: pinned("yeo-johnson", True), False),
    "bc_std1": (lambda: PowerTransformer("box-cox"), True),
    "bc_std0": (lambda: PowerTransformer("box-cox", standardize=False), True),
    "bc_pinned_std1": (lambda: pinned("box-cox", True), True),
}


def ordered(v):
    i = np.float64(v).view(np.int64).item()
    return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)


def run(name, make, positive, variant):
    make2 = make
    if positive:
        make2 = ct.positive(make)
    lanes, kmax, kmax_tight, umax = 0, 0.0, 0.0, 0
    ks, ks_t = [], []
    worst = None
    worst_ulp = None
    over = []
    for seed in range(SEEDS):
        try:
            step = ct._step(make2, seed, variant)
        except AssertionError:
            continue
        rows = ct._rows(step, seed, positive).to_pylist()
        for r in rows:
            iid = r["__iid"]
            if iid is None:
                continue
            est = step.instances[iid]
            vals = [math.nan if r[f] is None else float(r[f]) for f in step.takes.names]
            try:
                twin = np.asarray(est.transform([vals]), dtype=float)[0]
            except ValueError:
                continue  # the twin rejects the row
            for j, x in enumerate(vals):
                lam = float(est.lambdas_[j])
                t, w = (yj if est.method == "yeo-johnson" else bc)(x, lam)
                if est.method == "yeo-johnson":
                    if x >= 0 and abs(lam) >= EPS:
                        w = lam * float(np.log1p(x))
                    elif x < 0 and abs(lam - 2) > EPS:
                        w = (2.0 - lam) * float(np.log1p(-x))
                    else:
                        w = 0.0
                # twin's own pre-standardize value, for S
                if est.method == "yeo-johnson":
                    tt = float(stats.yeojohnson(np.array([x]), lam)[0])
                else:
                    from scipy.special import boxcox
                    tt = float(boxcox(x, lam))
                St = (1 + abs(w)) * abs(tt)
                St_tight = (1 + max(w, 0.0)) * abs(tt) if w == w else St
                nat = t
                if est.standardize:
                    m = float(est._scaler.mean_[j])
                    s = float(est._scaler.scale_[j])
                    nat = (t - m) / s
                    S = (St + abs(m)) / s
                    S_tight = (St_tight + abs(m)) / s
                else:
                    S, S_tight = St, St_tight
                a = float(twin[j])
                lanes += 1
                if (math.isnan(a) and math.isnan(nat)) or a == nat:
                    ks.append(0.0)
                    ks_t.append(0.0)
                    continue
                if math.isinf(a) or math.isinf(nat) or math.isnan(a) or math.isnan(nat):
                    over.append(dict(seed=seed, x=x, lam=lam, twin=a, native=nat))
                    continue
                d = abs(a - nat)
                k = d / (EPS * S) if S > 0 else math.inf
                kt = d / (EPS * S_tight) if S_tight > 0 else math.inf
                ks.append(k)
                ks_t.append(kt)
                u = abs(ordered(a) - ordered(nat))
                if k > kmax:
                    kmax = k
                    worst = dict(seed=seed, x=x, lam=lam, w=w, t_twin=tt, t_native=t,
                                 twin=a, native=nat, S=S, ulps=u)
                kmax_tight = max(kmax_tight, kt)
                if u > umax:
                    umax = u
                    worst_ulp = dict(seed=seed, x=x, lam=lam, w=w, t_twin=tt,
                                     t_native=t, twin=a, native=nat, S=S, K=k,
                                     mean=float(est._scaler.mean_[j]) if est.standardize else None,
                                     scale=float(est._scaler.scale_[j]) if est.standardize else None)
    ks = np.array(ks)
    ks_t = np.array(ks_t)
    res = dict(
        config=name, variant=variant, lanes=lanes, max_ulps=umax,
        K_max=kmax, K_p99=float(np.quantile(ks, 0.99)) if len(ks) else 0,
        K_p999=float(np.quantile(ks, 0.999)) if len(ks) else 0,
        K_tight_max=kmax_tight, frac_nonzero=float(np.mean(ks > 0)) if len(ks) else 0,
        inf_or_nan_mismatch=len(over), over_examples=over[:5],
        worst_K=worst, worst_ulp=worst_ulp,
    )
    print(json.dumps(res, default=float), flush=True)
    return res


results = []
for v, (name, (make, pos)) in enumerate(CONFIGS.items()):
    results.append(run(name, make, pos, 0))
json.dump(results, open(OUT, "w"), indent=1, default=float)
