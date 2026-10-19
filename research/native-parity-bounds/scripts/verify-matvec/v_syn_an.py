import pickle, sys, math, struct
import numpy as np
V = sys.argv[1]; EPS = 2.0**-52
data = pickle.load(open(f"{V}/syn.pkl", "rb"))
T = {ct: pickle.load(open(f"{V}/syn_{ct}.pkl", "rb")) for ct in ("SkylakeX", "Sandybridge", "Haswell")}
def split(a):
    t = a * 134217729.0; hi = t - (t - a); return hi, a - hi
def ordv(x):
    i, = struct.unpack("<q", struct.pack("<d", x)); return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
print(f"{'data':6s} {'n':>5s} | dot eps*S0: {'LTR-SKX':>7s} {'LTR-ex':>7s} {'SKX-ex':>7s} {'SNB-SKX':>7s} | lane K: {'e-SKX Srec':>10s} {'e-SKX S2':>8s} {'bat-row':>7s} {'SNB-SKX Srec':>12s} {'SNB-SKX S2':>10s} {'e-SNB S2':>8s} | {'maxK/sqrt n':>10s} {'max ulps e-SKX':>14s}")
for (name, n), d in data.items():
    est, X = d["est"], d["X"]; C = est.components_; m = est.mean_
    s = np.sqrt(est.explained_variance_); s[s < np.finfo(float).eps] = np.finfo(float).eps
    acc = X[:, :1] * C[:, 0]
    for i in range(1, n): acc = acc + X[:, i:i + 1] * C[:, i]
    mcS = T["SkylakeX"][(name, n)]["mc"]
    ent = (acc - mcS) / s
    S0 = np.abs(X) @ np.abs(C).T  # sum|x_i c_i| (rounding in S itself is negligible)
    Sm = np.abs(m) @ np.abs(C).T
    Srec = (S0 + np.abs(mcS)) / s; S2 = (S0 + Sm) / s
    exact = np.empty_like(acc)
    Ch, Cl = split(C)
    for r in range(X.shape[0]):
        P = X[r] * C; xh, xl = split(X[r])
        E = ((xh * Ch - P) + xh * Cl + xl * Ch) + xl * Cl
        for k in range(C.shape[0]): exact[r, k] = math.fsum(np.concatenate([P[k], E[k]]))
    sk = T["SkylakeX"][(name, n)]; sb = T["Sandybridge"][(name, n)]
    def dK(a, b): return np.max(np.abs(a - b) / (EPS * S0))
    def lK(a, b, S): return np.max(np.where(a == b, 0, np.abs(a - b) / (EPS * S)))
    Kmain = np.where(ent == sk["row"], 0, np.abs(ent - sk["row"]) / (EPS * Srec))
    mu = max(abs(ordv(float(a)) - ordv(float(b))) for a, b in zip(ent.ravel(), sk["row"].ravel()))
    print(f"{name:6s} {n:5d} | {'':11s} {dK(acc, sk['dot_row']):7.2f} {dK(acc, exact):7.2f} {dK(sk['dot_row'], exact):7.2f} {dK(sb['dot_row'], sk['dot_row']):7.2f} | {'':7s}"
          f" {lK(ent, sk['row'], Srec):10.2f} {lK(ent, sk['row'], S2):8.2f} {lK(sk['batch'], sk['row'], S2):7.2f} {lK(sb['row'], sk['row'], Srec):12.2f} {lK(sb['row'], sk['row'], S2):10.2f} {lK(ent, sb['row'], S2):8.2f} | {Kmain.max()/math.sqrt(n):10.3f} {mu:14.3g}")
