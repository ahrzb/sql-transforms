"""K for Yeo-Johnson over draws wider than the fixtures: lambda uniform in
[-5, 7] or at branch edges, x = +-loguniform[1e-300, 1e300] or uniform
(-1e3, 1e3). Twin: scipy.stats.yeojohnson on a 1-element array (as per row);
entry: kmeasure_lib.yj. usage: python stress.py N"""
import importlib.util, math, os, sys
import numpy as np
from scipy import stats
spec = importlib.util.spec_from_file_location("km", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kmeasure_lib.py"))
km = importlib.util.module_from_spec(spec); spec.loader.exec_module(km)
np.seterr(all="ignore")
EPS = 2.0**-52
N = int(sys.argv[1])
rng = np.random.default_rng(11)
edges = np.array([0.0, EPS / 2, EPS, -EPS, 2.0, 2 - EPS, 2 + 2 * EPS, 1.0, 1e-10, 2 - 1e-10])
lam = np.where(rng.random(N) < 0.2, rng.choice(edges, N), rng.uniform(-5, 7, N))
mag = np.where(rng.random(N) < 0.5, np.exp(rng.uniform(np.log(1e-300), np.log(1e300), N)), rng.uniform(0, 1e3, N))
x = mag * rng.choice([-1.0, 1.0], N)
Ks, Kt, over, nz = [], [], [], 0
for xi, li in zip(x.tolist(), lam.tolist()):
    tw = float(stats.yeojohnson(np.array([xi]), li)[0])
    nt, _ = km.yj(xi, li)
    if tw == nt or (math.isnan(tw) and math.isnan(nt)):
        Ks.append(0.0); Kt.append(0.0); continue
    if not (math.isfinite(tw) and math.isfinite(nt)):
        over.append((xi, li, tw, nt)); continue
    nz += 1
    if xi >= 0:
        w = li * math.log1p(xi) if abs(li) >= EPS else 0.0
    else:
        w = (2 - li) * math.log1p(-xi) if abs(li - 2) > EPS else 0.0
    d = abs(tw - nt)
    Ks.append(d / (EPS * (1 + abs(w)) * abs(tw)))
    if Ks[-1] > 2.5: print('lane', xi, li, w, tw, nt, Ks[-1])
    Kt.append(d / (EPS * (1 + max(w, 0.0)) * abs(tw)))
Ks, Kt = np.array(Ks), np.array(Kt)
WL = None
i = int(np.argmax(Kt))
print(f"lanes {N}: differ {nz} ({nz/N:.2%}), K max {Ks.max():.3f} p99.9 {np.quantile(Ks,0.999):.3f};"
      f" tight-S K max {Kt.max():.3f}; inf/nan mismatches {len(over)} {over[:3]}")
