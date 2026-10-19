import sys, math
sys.path.insert(0, sys.argv[1])
import numpy as np
import emul
from emul import fma
rng = np.random.default_rng(1)
# alternative tails
def tailB(ps, y):
    if len(ps) == 1: return fma(ps[0][0], ps[0][1], y)
    if len(ps) == 2: return y + fma(ps[0][0], ps[0][1], ps[1][0]*ps[1][1])
    t = fma(ps[0][0], ps[0][1], ps[1][0]*ps[1][1]); t = fma(ps[2][0], ps[2][1], t); return y + t
def tailC(ps, y):  # plain y + a*x for m3==1 too
    t = ps[0][0]*ps[0][1]
    for a,x in ps[1:]: t = fma(a,x,t)
    return y + t
rows=[]
for trial in range(400):
    m = int(rng.integers(1, 70)) if trial < 300 else int(rng.integers(2040, 2060))
    k = int(rng.integers(1, 12))
    C = rng.standard_normal((k, m)) * np.exp(rng.uniform(-3, 3, (k, m)))
    x = rng.standard_normal(m) * np.exp(rng.uniform(-3, 3, m))
    y = (x[None, :] @ C.T)[0].tolist()
    e = emul.gemv_t(x.tolist(), C.tolist())
    k4 = 4*(k//4)
    for j in range(k):
        if e[j] == y[j]: continue
        cls = "4x4" if j < k4 else ("4x2" if (j < k4 + 2 and k & 2) else "4x1")
        a = C[j].tolist(); xl = x.tolist(); m3 = m & 3; body = m - m3
        # body via emulation pieces
        if body >= 4:
            if m <= 2048 or True:
                # recompute body sum as emul does (single or two chunks)
                pass
        e0 = emul.gemv_t(xl[:body], [a[:body]] if cls=='4x1' else C[:, :body].tolist())
        b = e0[0] if cls=='4x1' else e0[j]
        ps = [(a[body+q], xl[body+q]) for q in range(m3)]
        alts = {'B': tailB(ps, b) if m3 else b, 'C': tailC(ps, b) if m3 else b}
        print(cls, m, k, j, 'numpy', y[j], 'emul', e[j], {kk: v == y[j] for kk, v in alts.items()})
