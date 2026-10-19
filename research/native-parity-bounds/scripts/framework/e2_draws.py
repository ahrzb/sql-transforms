"""E2/E3 inputs: chi2 draws (as the chi2 record) and Yeo-Johnson (x, lambda) draws."""
import numpy as np
rng = np.random.default_rng(20261006)
n = 200_000
chi2_x = np.concatenate([rng.uniform(0, 1e3, n)[1:], np.exp(rng.uniform(np.log(1e-300), np.log(1e300), n)),
                         1 + rng.normal(0, 1e-3, n)])
chi2_x = chi2_x[chi2_x > 0]
lam = np.concatenate([rng.uniform(-2, 4, 300), [0.0, 2.0, 1.0, 0.5, -1.0]])
yj_x = np.concatenate([rng.uniform(-1e3, 1e3, 1000), rng.normal(0, 1, 1000),
                       np.exp(rng.uniform(-30, 30, 1000)) * rng.choice([-1, 1], 1000)])
np.savez("e2_draws.npz", chi2_x=chi2_x, lam=lam, yj_x=yj_x)
print(len(chi2_x), len(lam) * len(yj_x))
