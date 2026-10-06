import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from threadpoolctl import threadpool_info
from emul import numpy_row_matvec
arch = [d["architecture"] for d in threadpool_info() if d["internal_api"] == "openblas"][0]
rng = np.random.default_rng(2)
hit = tot = 0; miss = []
for trial in range(1500):
    m = int(rng.integers(1, 80)) if trial < 1200 else int(rng.integers(2030, 2100))
    k = int(rng.integers(1, 14))
    C = rng.standard_normal((k, m)) * np.exp(rng.uniform(-3, 3, (k, m)))
    x = rng.standard_normal(m) * np.exp(rng.uniform(-3, 3, m))
    y = (x[None, :] @ C.T)[0].tolist()
    e = numpy_row_matvec(x.tolist(), C.tolist(), arch)
    for j in range(k):
        tot += 1
        if e[j] == y[j]: hit += 1
        else: miss.append((m, k, j))
print(arch, f"bit-exact {hit}/{tot}", "misses:", miss[:10])
