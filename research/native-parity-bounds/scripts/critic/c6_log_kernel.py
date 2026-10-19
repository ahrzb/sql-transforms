"""Which float64 log does numpy 2.5.1 run here: SVML log8_ha or numpy's own
AVX512F (Tang) kernel? Tang defers to npy_log (glibc) on
(1-2^-4, 1+0x1.09p-4); SVML does not. So mismatches vs glibc inside that
interval fingerprint SVML. Also: chi2 K at sklearn's TRUE defaults
(sample_steps=3: s=0.4, j=1,2), on log-mismatch lanes at large |ln x|."""
import math

import numpy as np
from sklearn.kernel_approximation import AdditiveChi2Sampler

rng = np.random.default_rng(3)
try:
    from numpy.lib.introspect import opt_func_info
    info = opt_func_info(func_name="^log$", signature="float64")
    print("opt_func_info log:", info)
except Exception as e:  # noqa: BLE001
    print("opt_func_info unavailable:", e)

x = 1.0 + rng.uniform(-0.06, 0.06, 1_000_000)
npl = np.log(x)
gl = np.array([math.log(v) for v in x])
print("near-1 (|x-1|<0.06) np.log != glibc:", int((npl != gl).sum()), "of", x.size)

# chi2 at the true sample_steps=3 default, enriched on log mismatches with large |ln x|
m = np.ldexp(rng.uniform(1, 2, 4_000_000), rng.integers(-1000, 1000, 4_000_000))
L_np = np.log(m)
L_gl = np.array([math.log(v) for v in m])
xs = m[L_np != L_gl]
print("enriched mismatch x:", xs.size)
eps = 2.0**-52
for steps, s in [(2, 0.5), (3, 0.4)]:
    est = AdditiveChi2Sampler(sample_steps=steps).fit(np.ones((2, 1)))
    assert est.sample_interval_ == s if hasattr(est, "sample_interval_") else True
    tw = np.vstack([est.transform(np.array([[v]]))[0] for v in xs])
    worst, cnt = 0.0, 0
    for i, v in enumerate(xs):
        L = math.log(v)
        lanes = [math.sqrt(v * s)]
        for j in range(1, steps):
            f = math.sqrt((2 * v * s) / float(np.cosh(np.pi * j * s)))
            a = j * (s * L)
            lanes += [f * math.cos(a), f * math.sin(a)]
        for k in range(1, len(lanes)):
            j = (k + 1) // 2
            f = math.sqrt((2 * v * s) / float(np.cosh(np.pi * j * s)))
            S = f * (1 + abs(j * s * L))
            K = abs(lanes[k] - tw[i][k]) / (eps * S)
            worst = max(worst, K)
            cnt += K > 1
    print(f"steps={steps} s={s}: lanes compared {len(xs) * (2 * steps - 2)}, max K {worst:.4f}, lanes K>1: {cnt}")
