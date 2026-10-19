import math, numpy as np
EPS = 2.0 ** -52; rng = np.random.default_rng(2024)
for n in [2048, 8192]:
    m = rng.uniform(1, 200, n); C = rng.uniform(0, 1, (2, n)); C /= np.linalg.norm(C, axis=1, keepdims=True)
    M = (m.reshape(1, -1) @ C.T)[0]
    rows = 300; X = m + rng.normal(size=(rows, n)) * 10.0 ** rng.uniform(-7, 0, (rows, 1))
    Ks = []
    for x in X:
        t = (x.reshape(1, -1) @ C.T)[0] - M
        for j in range(2):
            p = x * C[j]; acc = p[0]
            for v in p[1:].tolist(): acc = acc + v
            S = math.fsum(np.abs(p).tolist()) + abs(M[j])
            Ks.append(abs((acc - M[j]) - t[j]) / (EPS * S))
    Ks = np.array(Ks); print(f"n={n}: max K {Ks.max():.2f}, p99 {np.percentile(Ks, 99):.2f}, max/sqrt(n) {Ks.max()/math.sqrt(n):.3f}")
