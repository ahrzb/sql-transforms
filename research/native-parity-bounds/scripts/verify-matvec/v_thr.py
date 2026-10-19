"""gemv (row path) and gemm (batch) above OpenBLAS's gemv threading threshold
(m*n >= 115200*4 = 460800 in interface/gemv.c v0.3.33), C- and F-ordered C."""
import sys, numpy as np
from threadpoolctl import threadpool_info
info = [d for d in threadpool_info() if d["internal_api"] == "openblas"][0]
rng = np.random.default_rng(5); out = {}
for n, k in ((4096, 64), (32768, 18), (32768, 30), (100000, 5), (100000, 7), (20000, 30)):
    Cc = np.ascontiguousarray(rng.standard_normal((k, n)) / np.sqrt(n))
    X = rng.uniform(-100, 100, n) + rng.standard_normal((8, n))
    for lay, C in (("C", Cc), ("F", np.asfortranarray(Cc))):
        out[(n, k, lay, "row")] = np.array([(x[None, :] @ C.T)[0] for x in X])
        out[(n, k, lay, "batch")] = X @ C.T
np.save(sys.argv[1], out, allow_pickle=True); print(info["architecture"], info["num_threads"])
