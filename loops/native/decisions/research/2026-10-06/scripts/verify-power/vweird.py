import sys, math, struct, numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, sys.argv[1]); from vlib import yj_e  # noqa
EPS = 2.0 ** -52
def ordd(v):
    (i,) = struct.unpack("<q", struct.pack("<d", v)); return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
worst = 0; cnt = 0; hits = 0
for c in range(-3, 4):
    for lam in [0.0, EPS, -EPS, 2.0, 2 + 2 * EPS, 2 - EPS, 1.0, 0.5, -2.0, 3.0, 4.5, 1e-8]:
        for n in (3, 5, 7, 10, 17, 40):
            X = np.full((n, 1), float(c))
            t = stats.yeojohnson(X[:, 0].copy(), lam)
            sc = StandardScaler().fit(t.reshape(-1, 1)); m, s = float(sc.mean_[0]), float(sc.scale_[0])
            twin = float((stats.yeojohnson(np.array([float(c)]), lam)[0] - m) / s)
            tn = yj_e(float(c), lam); nat = (tn - m) / s
            tt = float(stats.yeojohnson(np.array([float(c)]), lam)[0])
            if c >= 0:
                w = 0.0 if abs(lam) < EPS else lam * float(np.log1p(c))
            else:
                w = 0.0 if abs(lam - 2) <= EPS else (2 - lam) * float(np.log1p(-c))
            S = ((1 + max(w, 0)) * abs(tt) + abs(m)) / s
            cnt += 1
            if twin != nat:
                hits += 1
                K = abs(twin - nat) / (EPS * S)
                if K > worst: worst = K
                if (c, lam, n) == (3, 2 + 2 * EPS, n) or abs(ordd(twin) - ordd(nat)) > 1e15:
                    print(f"c={c} lam={lam!r} n={n} mean_exact={m == tt} twin={twin!r} nat={nat!r} ulps={abs(ordd(twin)-ordd(nat)):.3g} K={K:.3f}")
print("lanes", cnt, "differing", hits, "worst K", worst)
