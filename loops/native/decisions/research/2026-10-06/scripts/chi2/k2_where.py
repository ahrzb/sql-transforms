"""Where numpy's log (SVML log8_ha) parts from glibc's: by x, 2,000,000 draws
log-uniform in x over (e^-8, e^8), plus the uniform(0,1e3) mismatches."""
import math, numpy as np
rng = np.random.default_rng(7)
x = rng.uniform(0.0, 1e3, 200_000)
L = np.array([math.log(v) for v in x.tolist()]); d = np.log(x) != L
print("uniform(0,1e3) mismatches:", d.sum(), "their x range", x[d].min(), x[d].max(), "L range", L[d].min(), L[d].max())
print("  sorted x of mismatches:", np.sort(x[d])[:50])
N = 2_000_000
x = np.exp(rng.uniform(-8, 8, N))
L = np.array([math.log(v) for v in x.tolist()]); d = np.log(x) != L
edges = np.array([-8,-4,-2,-1.5,-1,-0.5,-0.25,-0.0625,0,0.0625,0.25,0.5,1,1.5,2,4,8])
idx = np.digitize(L, edges)
for k in range(1, len(edges)):
    m = idx == k
    print(f"L in [{edges[k-1]:+.4g},{edges[k]:+.4g}): n={m.sum():7d} differ={d[m].sum():6d} rate={d[m].mean():.2e}")
xm = x[d]
print("mismatch x range:", xm.min(), xm.max())
