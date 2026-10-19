"""E2b: chi2 under the pinned (baseline-numpy) twin against the entry spelled
on glibc (math.log/cos/sin/cosh): bit-exact?"""
import math, numpy as np
d = np.load("e2_draws.npz"); tw = np.load("tw2_V2.npz"); x = d["chi2_x"].tolist()
tot = diff = 0
for s, js in [(0.8, [1]), (0.5, [1, 2]), (0.4, [1, 2, 3])]:
    for j in js:
        C = math.cosh(math.pi * j * s)
        for trig, fn in [("cos", math.cos), ("sin", math.sin)]:
            nat = np.array([math.sqrt((2 * v * s) / C) * fn(j * (s * math.log(v))) for v in x])
            t = tw[f"{trig}_{s}_{j}"]; tot += len(t); diff += int(np.sum(t != nat))
print(f"chi2, twin with numpy baseline kernels vs entry on glibc: lanes {tot}, differ {diff}")
