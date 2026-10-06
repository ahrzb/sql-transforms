import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import emul
def ddot_hsw_plain(x, y):
    n = len(x); n1 = n & -16
    d = emul.ddot_hsw(x[:n1], y[:n1]) if n1 else 0.0
    for i in range(n1, n): d = d + y[i]*x[i]
    return d
rng = np.random.default_rng(3); h1=h2=t=0
for trial in range(1500):
    m = int(rng.integers(1, 80))
    C = rng.standard_normal((1, m)) * np.exp(rng.uniform(-3, 3, (1, m)))
    x = rng.standard_normal(m) * np.exp(rng.uniform(-3, 3, m))
    y = (x[None, :] @ C.T)[0, 0]
    t += 1; h1 += emul.ddot_hsw(x.tolist(), C[0].tolist()) == y; h2 += ddot_hsw_plain(x.tolist(), C[0].tolist()) == y
print("haswell ddot, fma tail", h1, "/", t, "; plain tail", h2, "/", t)
