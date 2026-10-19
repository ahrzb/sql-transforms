"""Local search for large K near the worst stress lanes (twin: SVML numpy; entry: glibc via math)."""
import sys, math, numpy as np
sys.argv = [sys.argv[0], "none", sys.argv[1], sys.argv[2]]
sys.path.insert(0, sys.argv[3])
np.seterr(all="ignore"); EPS = 2.0 ** -52
import importlib.util
spec = importlib.util.spec_from_file_location("vlib", sys.argv[3] + "/vlib.py"); vl = importlib.util.module_from_spec(spec); spec.loader.exec_module(vl)
def yj_twin(x, lam):
    out = np.empty_like(x); w = np.zeros_like(x)
    pos = x >= 0
    a = pos & (np.abs(lam) < EPS); out[a] = np.log1p(x[a])
    b = pos & ~a; w[b] = lam[b] * np.log1p(x[b]); out[b] = np.expm1(w[b]) / lam[b]
    c = ~pos & (np.abs(lam - 2) > EPS); d2 = 2 - lam[c]; w[c] = d2 * np.log1p(-x[c]); out[c] = -np.expm1(w[c]) / d2
    e = ~pos & ~(np.abs(lam - 2) > EPS); out[e] = -np.log1p(-x[e])
    return out, w
ent = np.frompyfunc(vl.yj_e, 2, 1)
def K(x, lam):
    t, w = yj_twin(x, lam); n = ent(x, lam).astype(float)
    same = (t == n)
    return np.where(same, 0.0, np.abs(t - n) / (EPS * (1 + np.maximum(w, 0)) * np.abs(t))), t, n
z = np.load(sys.argv[2]); x0, l0 = z["x"], z["lam"]
k0, _, _ = K(x0[:2000000], l0[:2000000])
top = np.argsort(k0)[-150:]
print("start max", k0[top[-1]])
rng = np.random.default_rng(5)
best = (0, None)
for it in range(3):
    xs, ls = [], []
    for i in top:
        xi, li = x0[i], l0[i]
        steps = rng.integers(-2000, 2000, 3000)
        xs.append(xi + steps * np.spacing(xi)); ls.append(li * (1 + rng.uniform(-1e-3, 1e-3, 3000)))
    xs, ls = np.concatenate(xs), np.concatenate(ls)
    k, t, n = K(xs, ls)
    j = np.argmax(k)
    if k[j] > best[0]:
        best = (k[j], (repr(xs[j]), repr(ls[j]), repr(t[j]), repr(n[j])))
    order = np.argsort(k)[-150:]
    x0, l0 = xs, ls; top = order
    print("iter", it, "max K", k[j], "p99.99", np.quantile(k, 0.9999), flush=True)
print("best", best)
