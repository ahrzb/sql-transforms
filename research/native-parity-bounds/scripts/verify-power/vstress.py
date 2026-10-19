"""Stress: twin (scipy YJ ops, numpy kernels of this process) vs entry (Goldberg/Kahan on
numpy log/exp, which equal glibc when X86_V4 is disabled).
usage: vstress.py twin N OUT.npz   (run with SVML)
       vstress.py entry IN.npz      (run with NPY_DISABLE_CPU_FEATURES=X86_V4)"""
import sys, numpy as np
np.seterr(all="ignore")
EPS = 2.0 ** -52

def yj_twin(x, lam):
    # scipy _yeojohnson_transform, elementwise (lam per element)
    out = np.empty_like(x); w = np.zeros_like(x)
    pos = x >= 0
    a = pos & (np.abs(lam) < EPS); out[a] = np.log1p(x[a])
    b = pos & ~(np.abs(lam) < EPS)
    w[b] = lam[b] * np.log1p(x[b]); out[b] = np.expm1(w[b]) / lam[b]
    c = ~pos & (np.abs(lam - 2) > EPS)
    d2 = 2 - lam[c]; w[c] = d2 * np.log1p(-x[c]); out[c] = -np.expm1(w[c]) / d2
    e = ~pos & ~(np.abs(lam - 2) > EPS); out[e] = -np.log1p(-x[e])
    return out, w

def log1p_g(z):
    u = 1.0 + z
    return np.where(u == 1.0, z, z * (np.log(u) / (u - 1.0)))

def expm1_k(w):
    u = np.exp(w)
    r = (u - 1.0) * (w / np.log(u))
    r = np.where(u - 1.0 == -1.0, -1.0, r)
    r = np.where(u == 1.0, w, r)
    r = np.where(np.isinf(u), np.inf, r)
    return r

def yj_entry(x, lam):
    out = np.empty_like(x)
    pos = x >= 0
    a = pos & (np.abs(lam) < EPS); out[a] = log1p_g(x[a])
    b = pos & ~(np.abs(lam) < EPS); out[b] = expm1_k(lam[b] * log1p_g(x[b])) / lam[b]
    c = ~pos & (np.abs(lam - 2) > EPS); d2 = 2 - lam[c]; out[c] = -expm1_k(d2 * log1p_g(-x[c])) / d2
    e = ~pos & ~(np.abs(lam - 2) > EPS); out[e] = -log1p_g(-x[e])
    return out

def draws(N, rng):
    k = N // 8
    sgn = lambda n: rng.choice([-1.0, 1.0], n)
    L, X = [], []
    # 1 broad
    L.append(rng.uniform(-5, 7, k)); X.append(sgn(k) * 10.0 ** rng.uniform(-300, 300, k))
    # 2 small x (small |w|), lam broad
    L.append(rng.uniform(-5, 7, k)); X.append(sgn(k) * 10.0 ** rng.uniform(-17, -2, k))
    # 3 lam near 0 / near 2
    lam3 = np.concatenate([sgn(k // 2) * 10.0 ** rng.uniform(-16, -2, k // 2), 2 + sgn(k - k // 2) * 10.0 ** rng.uniform(-16, -2, k - k // 2)])
    L.append(lam3); X.append(sgn(k) * 10.0 ** rng.uniform(-5, 5, k))
    # 4 w near overflow: pick target w in [600, 709.7827], lam, solve L
    wt = rng.uniform(600, 709.7827, k); lam4 = rng.uniform(1.0, 7.0, k)
    pos = rng.random(k) < 0.5
    Lx = np.where(pos, wt / lam4, wt / np.abs(2 - (2 - lam4)))
    lam4 = np.where(pos, lam4, 2 - lam4)  # mirror branch: 2-lam = old lam4 > 0 -> w>0
    x4 = np.expm1(Lx); x4 = np.where(pos, x4, -x4)
    L.append(lam4); X.append(x4)
    # 5 uniform fixtures-like
    L.append(rng.uniform(-3, 4, k)); X.append(rng.uniform(-1e3, 1e3, k))
    # 6 strongly negative w
    L.append(rng.uniform(-50, -1, k)); X.append(10.0 ** rng.uniform(0, 300, k))
    # 7 moderate w, x in [0.01, 100]
    L.append(rng.uniform(-10, 12, k)); X.append(sgn(k) * 10.0 ** rng.uniform(-2, 2, k))
    # 8 x mid range near powers-of-two boundaries of 1+x
    n8 = N - 7 * k
    L.append(rng.uniform(-3, 5, n8)); X.append(sgn(n8) * (2.0 ** rng.integers(-30, 30, n8)) * (1 + rng.uniform(-1e-6, 1e-6, n8)))
    return np.concatenate(X), np.concatenate(L)

if sys.argv[1] == "twin":
    N = int(sys.argv[2]); rng = np.random.default_rng(2026)
    x, lam = draws(N, rng)
    t, w = yj_twin(x, lam)
    np.savez(sys.argv[3], x=x, lam=lam, t=t, w=w)
else:
    z = np.load(sys.argv[2]); x, lam, t, w = z["x"], z["lam"], z["t"], z["w"]
    n = yj_entry(x, lam)
    same = (t == n) | (np.isnan(t) & np.isnan(n))
    fin = np.isfinite(t) & np.isfinite(n)
    infm = ~same & ~fin
    St = (1 + np.maximum(w, 0)) * np.abs(t)
    Sr = (1 + np.abs(w)) * np.abs(t)
    d = np.abs(t - n)
    K = np.where(same, 0, np.where(fin, d / (EPS * St), np.nan))
    Kr = np.where(same, 0, np.where(fin, d / (EPS * Sr), np.nan))
    print("lanes", len(x), "differ", (~same).mean(), "inf/nan mismatches", infm.sum())
    if infm.any():
        i = np.where(infm)[0][:5]; print(" examples", list(zip(x[i], lam[i], w[i], t[i], n[i])))
    print("K_tight max", np.nanmax(K), "p99.9 (all)", np.nanquantile(K, 0.999), "K_rec max", np.nanmax(Kr))
    i = np.nanargmax(K)
    print(" worst: x", repr(x[i]), "lam", repr(lam[i]), "w", w[i], "t", repr(t[i]), "n", repr(n[i]))
    # per regime
    k = len(x) // 8
    for r in range(8):
        sl = slice(r * k, (r + 1) * k if r < 7 else len(x))
        print(" regime", r + 1, "Kmax", np.nanmax(K[sl]), "differ", (~same[sl]).mean())
    # Goldberg-only error: log1p entry vs log1p numpy-svml? no: compare in eps of |t| for small w
