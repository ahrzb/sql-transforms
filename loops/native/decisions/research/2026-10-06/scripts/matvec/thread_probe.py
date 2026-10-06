"""Row-path gemv (1,n)@(n,k) under the caller's OPENBLAS_NUM_THREADS; saves results."""
import sys
import numpy as np
rng = np.random.default_rng(11)
out = {}
for n in (512, 2048, 4096):
    for k in (5, 6, 7, 18, 30, 64):
        C = np.ascontiguousarray(np.linalg.qr(rng.standard_normal((n, k)))[0].T)
        X = rng.uniform(-100, 100, n) + rng.standard_normal((50, n))
        out[(n, k)] = np.array([(x[None, :] @ C.T)[0] for x in X])
np.save(sys.argv[1], out, allow_pickle=True)
