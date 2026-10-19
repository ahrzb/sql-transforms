"""Targeted: small |w| (x in 1e-17..1e-2), and Box-Cox stress. twin mode under SVML, entry mode under X86_V4 disabled.
Box-Cox twin = scipy.special.boxcox (glibc in either mode)."""
import sys, numpy as np
from scipy.special import boxcox
sys.path.insert(0, sys.argv[-1])
from vstress import yj_twin, yj_entry, expm1_k
np.seterr(all="ignore"); EPS = 2.0 ** -52
if sys.argv[1] == "twin":
    N = int(sys.argv[2]); rng = np.random.default_rng(777)
    sg = lambda n: rng.choice([-1.0, 1.0], n)
    x = sg(N) * 10.0 ** rng.uniform(-17, -2, N)
    lam = np.concatenate([rng.uniform(-8, 10, N // 2), sg(N - N // 2) * 10.0 ** rng.uniform(-15, 1, N - N // 2)])
    t, w = yj_twin(x, lam)
    xb = np.concatenate([10.0 ** rng.uniform(-300, 300, N // 2), rng.uniform(1e-6, 1e3, N - N // 2)])
    lb = np.concatenate([rng.uniform(-5, 5, N // 2), sg(N - N // 2) * 10.0 ** rng.uniform(-19, 1, N - N // 2)])
    tb = boxcox(xb, lb)
    np.savez(sys.argv[3], x=x, lam=lam, t=t, w=w, xb=xb, lb=lb, tb=tb)
else:
    z = np.load(sys.argv[2])
    x, lam, t, w = z["x"], z["lam"], z["t"], z["w"]
    n = yj_entry(x, lam)
    same = (t == n) | (np.isnan(t) & np.isnan(n))
    K = np.where(same, 0, np.abs(t - n) / (EPS * (1 + np.maximum(w, 0)) * np.abs(t)))
    i = np.nanargmax(K)
    print("YJ small-w lanes", len(x), "differ", (~same).mean(), "Kmax", K[i], "p99.99", np.nanquantile(K, 0.9999), "x", repr(x[i]), "lam", repr(lam[i]), "w", w[i])
    xb, lb, tb = z["xb"], z["lb"], z["tb"]
    lx = np.log(xb)
    wb = lb * lx
    e = np.where(np.abs(lb) < 1e-19, lx, np.where(wb < 709.78, expm1_k(wb) / lb,
        np.copysign(1.0, lb) * np.exp(wb - np.log(np.abs(lb))) - 1 / lb))
    sm = (e == tb) | (np.isnan(e) & np.isnan(tb))
    fin = np.isfinite(e) & np.isfinite(tb)
    Kb = np.where(sm, 0, np.where(fin, np.abs(e - tb) / (EPS * np.abs(tb)), np.nan))
    ul = np.abs(e.view(np.int64) - tb.view(np.int64))
    print("BC lanes", len(xb), "differ", (~sm).mean(), "inf/nan mismatch", int((~sm & ~fin).sum()), "Kmax", np.nanmax(Kb), "max ulps", ul[fin].max())
