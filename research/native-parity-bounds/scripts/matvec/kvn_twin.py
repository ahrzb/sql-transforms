"""One BLAS configuration: per dataset, the row path (x[None,:] @ C.T, one row at a
time, = the twin's first op: gemv / ddot) and the batch path (X @ C.T, gemm), plus
the full PCA lane est.transform([row]) for the pca datasets. Also mc = mean @ C.T
as this configuration computes it."""

import pickle
import sys
import warnings

import numpy as np
from threadpoolctl import threadpool_info

warnings.simplefilter("ignore")
arch = [d["architecture"] for d in threadpool_info() if d["internal_api"] == "openblas"][0]
data = pickle.load(open(sys.argv[1], "rb"))
res = {}
for key, d in data.items():
    C, X = d["C"], d["X"]
    row = np.array([(x[None, :] @ C.T)[0] for x in X])
    batch = X @ C.T
    mc = (np.reshape(d["mean"], (1, -1)) @ C.T)[0]
    r = dict(row=row, batch=batch, mc=mc)
    if "est" in d:
        r["lane_row"] = np.array([d["est"].transform([x])[0] for x in X])
        r["lane_batch"] = d["est"].transform(X)
    res[key] = r
pickle.dump(res, open(sys.argv[2], "wb"))
print(arch, "done")
