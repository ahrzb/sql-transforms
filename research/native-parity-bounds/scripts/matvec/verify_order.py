"""Does the emulated dgemv_t order reproduce numpy's (1,m) @ (m,k) bit for bit?
Run under a given OPENBLAS_CORETYPE (env set by the caller)."""

import sys

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
from threadpoolctl import threadpool_info  # noqa: E402

from emul import gemv_t, ltr  # noqa: E402

arch = [d["architecture"] for d in threadpool_info() if d["internal_api"] == "openblas"][0]
rng = np.random.default_rng(1)
variants = {
    "haswell+fmatail": dict(haswell=True, tail="fma_chain"),
    "haswell+plaintail": dict(haswell=True, tail="plain"),
    "generic": dict(haswell=False, tail="plain"),
}
hits = {v: 0 for v in variants}
hits["ltr"] = 0
total = 0
for trial in range(400):
    m = int(rng.integers(1, 70)) if trial < 300 else int(rng.integers(2040, 2060))
    k = int(rng.integers(1, 12))
    C = rng.standard_normal((k, m)) * np.exp(rng.uniform(-3, 3, (k, m)))
    x = rng.standard_normal(m) * np.exp(rng.uniform(-3, 3, m))
    y = (x[None, :] @ C.T)[0]
    Cl = C.tolist()
    xl = x.tolist()
    for v, kw in variants.items():
        e = gemv_t(xl, Cl, **kw)
        hits[v] += sum(a == b for a, b in zip(e, y.tolist()))
    hits["ltr"] += sum(ltr(xl, c) == b for c, b in zip(Cl, y.tolist()))
    total += k
print(arch, {v: f"{h}/{total}" for v, h in hits.items()})
