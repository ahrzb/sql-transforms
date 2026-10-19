"""The record's 'pinned lambdas + standardize: twin 0.0 vs 8.9e-16' lane,
reconstructed: an int64 feature whose fit column is one integer c (catalog
kind 2, rounded), the scaler fit on the pinned transform (mean_ = t(c),
scale_ = 1), served x = c (the 'ints' regime). Twin: sklearn; entry: the
spelling of kmeasure.py. Reports ulps of the result and K = d/(eps*S)."""

import math
import typing
import warnings

_eval = typing._eval_type
typing._eval_type = lambda *a, prefer_fwd_module=None, **k: _eval(*a, **k)

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.preprocessing import PowerTransformer, StandardScaler  # noqa: E402

import importlib.util, os  # noqa: E401,E402

spec = importlib.util.spec_from_file_location(
    "km", os.path.join(os.path.dirname(__file__), "kmeasure_lib.py"))
km = importlib.util.module_from_spec(spec)
spec.loader.exec_module(km)

warnings.filterwarnings("ignore")
EPS = 2.0**-52
lams = km.YJ_LAMBDAS + [0.25, -1.0, 1.5, 3.0 + 2 * EPS]
rows = []
exact_mean = 0
total_mean = 0
for c in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0):
    for n in (5, 6, 7, 17, 31, 59):
        X = np.full((n, 1), c)
        for lam in lams:
            est = PowerTransformer(standardize=True).fit(X)
            est.lambdas_ = np.array([lam])
            Xt = stats.yeojohnson(X[:, 0], lam).reshape(-1, 1)
            est._scaler = StandardScaler(copy=False).set_output(
                transform="default").fit(Xt)
            m, s = float(est._scaler.mean_[0]), float(est._scaler.scale_[0])
            tt = float(Xt[0, 0])
            total_mean += 1
            exact_mean += m == tt
            twin = float(est.transform([[c]])[0, 0])
            t, w = km.yj(c, lam)
            nat = (t - m) / s
            if c >= 0 and abs(lam) >= EPS:
                w = lam * float(np.log1p(c))
            elif c < 0 and abs(lam - 2) > EPS:
                w = (2.0 - lam) * float(np.log1p(-c))
            else:
                w = 0.0
            S = ((1 + abs(w)) * abs(tt) + abs(m)) / s
            d = abs(nat - twin)
            rows.append((km.ordered_dist(twin, nat), d / (EPS * S), c, n, lam,
                         twin, nat, tt, t, m, s))
rows.sort(key=lambda r: -r[0])
print(f"mean_ == t(c) exactly in {exact_mean}/{total_mean} constant fits")
print("ulps        K      c   n  lambda                 twin      native     t_twin              t_native            mean_")
for r in rows[:12]:
    print(f"{r[0]:<10.3g} {r[1]:6.3f} {r[2]:4.0f} {r[3]:3d} {r[4]!r:22} {r[5]!r:10} {r[6]:<10.3g} {r[7]!r:19} {r[8]!r:19} {r[9]!r}")
print("max K over all:", max(r[1] for r in rows), "lanes:", len(rows),
      "lanes with twin 0.0 and native != 0:", sum(1 for r in rows if r[5] == 0 and r[6] != 0))
