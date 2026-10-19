import math, numpy as np
for s, j in [(0.8, 1), (0.5, 1), (0.5, 2), (0.4, 1), (0.4, 2), (0.4, 3)]:
    a = np.pi * j * s
    print(s, j, float(np.cosh(a)).hex(), math.cosh(a).hex(), float(np.cosh(np.array([a]))[0]).hex())
rng = np.random.default_rng(0); x = rng.uniform(-1e3, 1e3, 1_000_000)
c = np.cos(x); g = np.array([math.cos(v) for v in x.tolist()])
print("np.cos vs glibc cos differ:", int((c != g).sum()), "of", len(x))
s = np.sin(x); g = np.array([math.sin(v) for v in x.tolist()])
print("np.sin vs glibc sin differ:", int((s != g).sum()))
