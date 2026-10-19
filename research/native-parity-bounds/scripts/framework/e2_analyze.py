"""E2 (chi2) and E3 (Yeo-Johnson): native = glibc via Python math with the
entry's spellings; exact = Decimal at 60 digits on a subsample. Compares the
twin under numpy X86_V4 / X86_V3 / baseline kernels, native, and exact."""
import math, sys, numpy as np
from decimal import Decimal as D, getcontext
getcontext().prec = 60
EPS = 2.0**-52
d = np.load("e2_draws.npz"); V = {v: np.load(f"tw2_{v}.npz") for v in ["V4", "V3", "V2"]}

def ordered(a):
    i = np.asarray(a, np.float64).view(np.int64)
    return np.where(i >= 0, i, -(i & 0x7FFF_FFFF_FFFF_FFFF)).astype(object)

def ulps(a, b):
    return np.abs(ordered(a) - ordered(b)).astype(float)

# pi to 70 digits (decimal docs recipe)
def dpi():
    getcontext().prec += 5
    three = D(3); lasts, t, s, n, na, dd, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s; n, na = n + na, na + 8; dd, da = dd + da, da + 32; t = (t * n) / dd; s += t
    getcontext().prec -= 5
    return +s
PI = dpi()
def dcos_sin(th):
    r = th - (th / (2 * PI)).to_integral_value() * 2 * PI
    c = s = D(0); term = D(1); k = 0
    while True:  # sum r^k/k! split by parity
        if k % 4 == 0: c += term
        elif k % 4 == 1: s += term
        elif k % 4 == 2: c -= term
        else: s -= term
        k += 1; term = term * r / k
        if abs(term) < D(10) ** -65: break
    return c, s

out = []
def report(name, a, b, S):
    a, b, S = map(lambda z: np.asarray(z, float), (a, b, S))
    fin = np.isfinite(a) & np.isfinite(b)
    nf = int(np.sum(~fin & ~((a == b) | (np.isnan(a) & np.isnan(b)))))
    a, b, S = a[fin], b[fin], S[fin]
    u = ulps(a, b); K = np.abs(a - b) / (EPS * S)
    K[(a == b)] = 0.0
    line = (f"{name}: lanes {len(a)}, differ {int((a != b).sum())}, max ulps {u.max():.0f}, p99.9 ulps {np.percentile(u, 99.9):.0f},"
            f" max K {np.nanmax(K):.3f}, p99.9 K {np.nanpercentile(K, 99.9):.3f}, sign flips {int(np.sum(np.sign(a) != np.sign(b)))}, nonfinite mismatches {nf}")
    print(line); out.append(line)

# ---------------- chi2
x = d["chi2_x"]; xl = x.tolist()
glog = np.array([math.log(v) for v in xl])
for v in V:
    print(f"np.log[{v}] vs glibc log: differ on {int(np.sum(V[v]['log'] != glog))} of {len(x)}, max ulps {ulps(V[v]['log'], glog).max():.0f}")
sub = np.random.default_rng(1).choice(len(x), 30000, replace=False)
for s, js in [(0.8, [1]), (0.5, [1, 2]), (0.4, [1, 2, 3])]:
    for j in js:
        C = float(np.cosh(np.pi * j * s))
        ls = [s * L for L in glog.tolist()]
        f = [math.sqrt((2 * xi * s) / C) for xi in xl]
        th = [j * t for t in ls]
        nat = {"cos": np.array([fi * math.cos(t) for fi, t in zip(f, th)]),
               "sin": np.array([fi * math.sin(t) for fi, t in zip(f, th)])}
        S = np.array(f) * (1 + np.abs(np.array(th)))
        # exact on the subsample
        ex = {"cos": [], "sin": []}
        for i in sub:
            xd = D(float(x[i])); thd = D(j) * D(s) * xd.ln()
            fd = (2 * xd * D(s) / D(C)).sqrt(); c, sn = dcos_sin(thd)
            ex["cos"].append(fd * c); ex["sin"].append(fd * sn)
        for trig in ["cos", "sin"]:
            key = f"{trig}_{s}_{j}"
            report(f"chi2 {key} twin[V4] vs native", V["V4"][key], nat[trig], S)
            report(f"chi2 {key} twin[V4] vs twin[V2]", V["V4"][key], V["V2"][key], S)
            report(f"chi2 {key} twin[V3] vs twin[V2]", V["V3"][key], V["V2"][key], S)
            report(f"chi2 {key} twin[V2] vs native", V["V2"][key], nat[trig], S)
            for who, arr in [("twin[V4]", V["V4"][key]), ("native", nat[trig])]:
                errs = [abs(D(float(arr[i])) - e) for i, e in zip(sub, ex[trig])]
                K = np.array([float(e) for e in errs]) / (EPS * S[sub])
                ul = max(float(e) / math.ulp(float(arr[i])) if arr[i] != 0 else 0 for i, e in zip(sub, errs))
                line = f"chi2 {key} {who} vs exact (30k sub): max K {K.max():.3f}, p99.9 K {np.percentile(K, 99.9):.3f}, max err in ulps of own result {ul:.1f}"
                print(line); out.append(line)

# ---------------- Yeo-Johnson
lam = d["lam"].tolist(); y = d["yj_x"].tolist()
def log1p_g(z):  # Goldberg, from glibc log (the power record's spelling)
    u = 1.0 + z
    return z if u == 1.0 else z * (math.log(u) / (u - 1.0))
def expm1_k(w):  # Kahan, from glibc exp/log (power.py)
    if w > 709.78: return math.inf
    u = math.exp(w)
    if u == 1.0: return w
    if u - 1.0 == -1.0: return -1.0
    return (u - 1.0) * (w / math.log(u))
e = float(np.finfo(float).eps)
nat, S, W = [], [], []
for l in lam:
    for z in y:
        if z >= 0:
            if abs(l) < e: t = log1p_g(z); w = t
            else: w = l * log1p_g(z); t = expm1_k(w) / l
        else:
            if abs(l - 2) > e: w = (2 - l) * log1p_g(-z); t = -expm1_k(w) / (2 - l)
            else: t = -log1p_g(-z); w = t
        nat.append(t); W.append(w); S.append((1 + abs(w)) * abs(t))
nat = np.array(nat); S = np.array(S)
report("YJ twin[V4] vs native", V["V4"]["yj"], nat, S)
report("YJ twin[V4] vs twin[V2]", V["V4"]["yj"], V["V2"]["yj"], S)
report("YJ twin[V2] vs native", V["V2"]["yj"], nat, S)
g1 = np.array([math.log1p(v) for v in np.abs(np.array(y)).tolist()])
for v in V:
    print(f"np.log1p[{v}] vs glibc: differ {int(np.sum(V[v]['log1p'] != g1))}/{len(g1)}, max ulps {ulps(V[v]['log1p'], g1).max():.0f}")
wv = np.linspace(-40, 700, 200001); ge = np.array([math.expm1(v) for v in wv.tolist()])
for v in V:
    print(f"np.expm1[{v}] vs glibc: differ {int(np.sum(V[v]['expm1'] != ge))}/{len(ge)}, max ulps {ulps(V[v]['expm1'], ge).max():.0f}")
# exact on 30k subsample
idx = np.random.default_rng(2).choice(len(nat), 30000, replace=False)
for who, arr in [("twin[V4]", V["V4"]["yj"]), ("twin[V2]", V["V2"]["yj"]), ("native", nat)]:
    Ks = []; worst_ulps = 0.0
    for i in idx:
        l = D(lam[i // len(y)]); z = D(y[i % len(y)])
        lf = lam[i // len(y)]
        if y[i % len(y)] >= 0:
            ex = (D(1) + z).ln() if abs(lf) < e else ((l * (D(1) + z).ln()).exp() - 1) / l
        else:
            ex = -((D(1) - z).ln()) if abs(lf - 2) <= e else -(((D(2) - l) * (D(1) - z).ln()).exp() - 1) / (D(2) - l)
        a = float(arr[i])
        if not math.isfinite(a) or abs(ex) > D("1.7e308"):
            continue
        err = float(abs(D(a) - ex)); Ks.append(err / (EPS * S[i]) if S[i] > 0 else 0.0)
        if a != 0: worst_ulps = max(worst_ulps, err / math.ulp(a))
    Ks = np.array(Ks)
    line = f"YJ {who} vs exact (30k sub): max K {Ks.max():.3f}, p99.9 K {np.percentile(Ks, 99.9):.3f}, max err in ulps of own result {worst_ulps:.1f}"
    print(line); out.append(line)
open("e2_out.txt", "w").write("\n".join(out))
