"""Does glibc's own log/cos/sin change between its FMA and SSE2 ifunc variants?
Writes the outputs' hash and the outputs to compare across processes."""
import math, sys, hashlib, numpy as np
rng = np.random.default_rng(9)
x = np.concatenate([rng.uniform(0, 1e3, 500_000), np.ldexp(rng.uniform(1, 2, 500_000), rng.integers(-1000, 1000, 500_000)), 1 + rng.normal(0, 1e-3, 500_000)])
x = x[x > 0]
a = rng.uniform(-1e3, 1e3, 500_000)
L = np.array([math.log(v) for v in x.tolist()])
C = np.array([math.cos(v) for v in a.tolist()])
Sn = np.array([math.sin(v) for v in a.tolist()])
np.save(sys.argv[1], np.concatenate([L, C, Sn]))
