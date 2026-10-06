"""Synthetic 1-ulp model: L' = L +- 1 ulp, sklearn's operation order, glibc
cos/sin (numpy's float64 cos/sin are libm). Report sup A and sup K per (s, j)."""
import numpy as np

EPS = 2.0**-52
rng = np.random.default_rng(11)
N = 2_000_000


def run(s, j, lo=1.0, hi=700.0):
    # random L with random mantissa in [lo, hi) and either sign
    L = rng.uniform(lo, hi, N)
    L = (L.view(np.uint64) ^ rng.integers(0, 2**40, N, dtype=np.uint64)).view(np.float64)
    L *= rng.choice([-1.0, 1.0], N)
    Lp = np.nextafter(L, np.where(rng.random(N) < 0.5, np.inf, -np.inf))
    t, tp = s * L, s * Lp
    a, ap = j * t, j * tp
    A = np.abs(ap - a) / (EPS * j * s * np.abs(L))
    x = np.exp(L)  # only for f; f identical on both sides anyway
    f = np.sqrt((2 * x * s) / np.cosh(np.pi * j * s))
    ok = np.isfinite(f) & (f > 0)
    S = EPS * f * (1 + np.abs(a))
    Kc = np.abs(f * np.cos(ap) - f * np.cos(a)) / S
    Ks = np.abs(f * np.sin(ap) - f * np.sin(a)) / S
    K = np.maximum(Kc, Ks)[ok]
    return A.max(), K.max()


cases = [(0.5, 1), (0.4, 1), (0.4, 2), (0.4, 3), (0.26, 3), (0.2501, 1), (0.2501, 3),
         (0.2501, 5), (0.2501, 9), (0.2501, 17), (0.2501, 33), (0.26, 17), (0.3, 3), (0.8, 3), (0.5000001, 3)]
for s, j in cases:
    A, K = run(s, j)
    print(f"s={s:<10} j={j:<3} supA={A:.4f} supK={K:.4f}")
