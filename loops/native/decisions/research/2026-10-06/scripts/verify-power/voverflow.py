import sys, math, numpy as np
from scipy import stats
sys.path.insert(0, sys.argv[1]); from vlib import yj_e
np.seterr(all="ignore")
EPS = 2.0 ** -52; DMAX = sys.float_info.max
# find a lane with twin != entry and a large t, w far from ln(DBL_MAX)
rng = np.random.default_rng(1)
for x in 10.0 ** rng.uniform(250, 300, 2000):
    lam = 1.0 + rng.uniform(-0.02, 0.02)
    tt = float(stats.yeojohnson(np.array([x]), lam)[0]); tn = yj_e(float(x), lam)
    if tt != tn and math.isfinite(tt):
        w = lam * float(np.log1p(x))
        break
print("x", x, "lam", lam, "w", w, "twin t", repr(tt), "entry t", repr(tn))
lo, hi = min(tt, tn), max(tt, tn)
# a scale_ that puts DBL_MAX between them: lo/s finite, hi/s inf (m = 0)
s = float(((lo + hi) / 2) / DMAX)
zt, zn = tt / s, tn / s
print("scale_", s, "twin z", zt, "entry z", zn)
K = 10.0
print("report's rule passes?", abs(w - math.log(DMAX)) <= K * EPS * (1 + abs(w)), "|w - ln DBL_MAX| =", abs(w - math.log(DMAX)))
S = ((1 + max(w, 0)) * abs(tt)) / s
print("finite side within K*eps*S of DBL_MAX?", DMAX - min(abs(zt), abs(zn)) <= K * EPS * S if math.isfinite(S) else "S overflows", "S=", S)
