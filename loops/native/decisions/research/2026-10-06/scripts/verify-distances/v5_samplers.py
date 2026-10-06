"""RBFSampler / SkewedChi2Sampler: twin = est.transform([row])[0]; native = ltr dot (+b), glibc cos, *c.
S_term = c(sum|L_i W_ij| + |b_j| + 1); S_mag = c(|theta_j| + 1)."""
import math, numpy as np
from sklearn.kernel_approximation import RBFSampler, SkewedChi2Sampler
EPS = 2.0 ** -52
rng = np.random.default_rng(31337)
gcos = np.frompyfunc(math.cos, 1, 1); glog = np.frompyfunc(math.log, 1, 1)
pool = {"rbf": [], "sk": []}; poolmag = {"rbf": [], "sk": []}; cosdiff = [0, 0]; logdiff = [0, 0]
def native(L, W, b, c):
    acc = L[0] * W[0]
    for i in range(1, len(L)):
        acc = acc + L[i] * W[i]
    th = acc + b
    co = gcos(th).astype(float)
    nc = np.cos(th); cosdiff[0] += int((nc != co).sum()); cosdiff[1] += th.size
    return co * c, th
def run(est, rows, fam, label):
    W, b, D = est.random_weights_, est.random_offset_, est.n_components
    c = (2.0 / D) ** 0.5 if fam == "rbf" else np.sqrt(2.0) / np.sqrt(D)
    Ks, Km = [], []
    for r in rows:
        x = [float(v) for v in r]
        tw = est.transform([x])[0]
        if fam == "sk":
            L = np.array(glog(np.array(x) + est.skewedness), dtype=float)
            Ln = np.log(np.array(x) + est.skewedness); logdiff[0] += int((Ln != L).sum()); logdiff[1] += L.size
        else:
            L = np.array(x)
        nat, th = native(L, W, b, c)
        S = c * (np.abs(L[:, None] * W).sum(axis=0) + np.abs(b) + 1)
        Ks.append(np.abs(nat - tw) / (EPS * S)); Km.append(np.abs(nat - tw) / (EPS * c * (np.abs(th) + 1)))
    Ks, Km = np.concatenate(Ks), np.concatenate(Km)
    pool[fam].append(Ks); poolmag[fam].append(Km)
    print(f"{label:<44} lanes={Ks.size:>7} K_term max={Ks.max():.3f} p99.9={np.percentile(Ks,99.9):.3f} | K_mag max={Km.max():.3g}", flush=True)
for n in (5, 20, 100):
    for g, off in ((0.05, 0.0), (2.0, 0.0), (0.05, 300.0), (None, 30.0)):
        X = rng.standard_t(4, size=(800, n)) * 3 + off
        est = RBFSampler(gamma=("scale" if g is None else g), n_components=300, random_state=n).fit(X)
        run(est, X[:400 if n < 100 else 150], "rbf", f"RBF n={n} gamma={g} off={off}")
    for s, kind in ((1.0, "lognormal"), (0.001, "expo"), (5.0, "unif0-1e3")):
        if kind == "lognormal": X = rng.lognormal(0, 2, size=(800, n))
        elif kind == "expo": X = rng.exponential(1.0, size=(800, n)) * (rng.uniform(size=(800, n)) > 0.3)
        else: X = rng.uniform(0, 1e3, size=(800, n))
        est = SkewedChi2Sampler(skewedness=s, n_components=300, random_state=n).fit(X)
        run(est, X[:400 if n < 100 else 150], "sk", f"SkewedChi2 n={n} s={s} {kind}")
for f in pool:
    a = np.concatenate(pool[f]); m = np.concatenate(poolmag[f])
    print(f"POOLED {f}: lanes={a.size} K_term max={a.max():.3f} | K_mag max={m.max():.3g}")
print(f"np.cos != glibc cos: {cosdiff[0]}/{cosdiff[1]} ; np.log != glibc log: {logdiff[0]}/{logdiff[1]}")
