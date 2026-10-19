"""Own driver for the report's emulator: new shapes (n up to 6200, k up to 22,
every n mod 4, cancelling data), C-contiguous components."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/matvec")
import numpy as np
from threadpoolctl import threadpool_info
from emul import numpy_row_matvec
arch = [d["architecture"] for d in threadpool_info() if d["internal_api"] == "openblas"][0]
rng = np.random.default_rng(int(sys.argv[1]))
res = {}
for trial in range(600):
    band = trial % 4
    n = [int(rng.integers(1, 40)), int(rng.integers(100, 600)), int(rng.integers(2040, 2060)), int(rng.integers(4090, 6200))][band]
    k = int(rng.integers(1, 23))
    if band == 3 and k * n >= 460800: k = max(1, 460800 // n - 1)
    mu = rng.uniform(-1e3, 1e3, n)
    C = rng.standard_normal((k, n)); x = mu + 1e-3 * rng.standard_normal(n)
    y = (x[None, :] @ C.T)[0].tolist()
    e = numpy_row_matvec(x.tolist(), C.tolist(), arch)
    for j in range(k):
        h, t = res.get((band, k == 1), (0, 0)); res[(band, k == 1)] = (h + (e[j] == y[j]), t + 1)
for key in sorted(res): print(arch, "n-band", ["1-39", "100-599", "2040-2059", "4090-6199"][key[0]], "k==1" if key[1] else "k>=2", "bit-exact %d/%d" % res[key])
