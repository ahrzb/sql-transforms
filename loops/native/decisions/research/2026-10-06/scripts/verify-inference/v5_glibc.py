import sys, numpy as np, scipy.special as sp
rng = np.random.default_rng(99)
x = np.concatenate([rng.lognormal(0, 3, 300000), np.arange(1, 5001, dtype=float), np.round(rng.lognormal(3.5, 1.2, 200000), 2) + 0.01])
lam = rng.uniform(-2, 2, len(x))
np.savez(sys.argv[1], bc=sp.boxcox(x, lam), log=np.log(x) if False else np.array([__import__('math').log(v) for v in x]),
         expm1=np.array([__import__('math').expm1(v) for v in lam * np.log(x)]))
