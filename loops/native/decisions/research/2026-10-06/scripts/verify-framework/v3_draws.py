import numpy as np
rng = np.random.default_rng(424242)
n = 150_000
x = np.concatenate([rng.uniform(0, 1e3, n), rng.uniform(0, 1e6, n), np.exp(rng.uniform(np.log(1e-300), np.log(1e300), n)),
                    1 + rng.normal(0, 1e-3, n), rng.lognormal(0, 3, n)])
x = x[x > 0]
np.save("v3_x.npy", x); print(len(x))
