import sys
from collections import Counter

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402

from emul import gemv_t  # noqa: E402

rng = np.random.default_rng(1)
bad = Counter()
tot = Counter()
for trial in range(400):
    m = int(rng.integers(1, 70)) if trial < 300 else int(rng.integers(2040, 2060))
    k = int(rng.integers(1, 12))
    C = rng.standard_normal((k, m)) * np.exp(rng.uniform(-3, 3, (k, m)))
    x = rng.standard_normal(m) * np.exp(rng.uniform(-3, 3, m))
    y = (x[None, :] @ C.T)[0].tolist()
    e = gemv_t(x.tolist(), C.tolist(), haswell=True, tail="fma_chain")
    k4 = 4 * (k // 4)
    for j in range(k):
        cls = "4x4" if j < k4 else ("4x2" if (j < k4 + 2 and k & 2) else "4x1")
        key = (cls, m & 3, m >= 2048)
        tot[key] += 1
        if e[j] != y[j]:
            bad[key] += 1
for key in sorted(tot):
    print(key, bad[key], "/", tot[key])
