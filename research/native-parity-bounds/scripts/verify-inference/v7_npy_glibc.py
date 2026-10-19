import math, numpy as np, json, os
rng = np.random.default_rng(8080)
D = {"log": np.concatenate([2.0 ** rng.uniform(-1020, 1020, 300000), rng.uniform(0.5, 2, 200000), np.arange(1, 100001.)]),
     "log1p": np.concatenate([2.0 ** rng.uniform(-60, 1000, 200000), rng.uniform(-0.999, 3, 300000)]),
     "exp": rng.uniform(-745, 709, 500000), "expm1": np.concatenate([rng.uniform(-40, 709, 250000), rng.uniform(-1, 1, 250000)]),
     "cos": np.concatenate([rng.uniform(-10, 10, 300000), rng.uniform(-1e5, 1e5, 200000)]),
     "sin": np.concatenate([rng.uniform(-10, 10, 300000), rng.uniform(-1e5, 1e5, 200000)]),
     "cosh": rng.uniform(-20, 20, 300000)}
out = {}
for f, x in D.items():
    a = getattr(np, f)(x); b = np.array([getattr(math, f)(v) for v in x.tolist()])
    out[f] = int((a != b).sum())
print(os.environ.get("NPY_DISABLE_CPU_FEATURES", "default"), json.dumps(out))
