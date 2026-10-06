"""One twin configuration (BLAS env set by the caller): transform every valid
row of the pickled fixtures one row at a time, as PythonTransform.__call__
does (`est.transform([vals])`), and as one batch per instance
(`est.transform(X)`), and save the lanes in a fixed order."""

import pickle
import sys
import warnings

import numpy as np
from threadpoolctl import threadpool_info

fx_path, out_path = sys.argv[1], sys.argv[2]
arch = [d["architecture"] for d in threadpool_info() if d["internal_api"] == "openblas"][0]
nth = [d["num_threads"] for d in threadpool_info() if d["internal_api"] == "openblas"][0]
with open(fx_path, "rb") as f:
    fixtures = pickle.load(f)

row_lanes, batch_lanes = [], []
warnings.simplefilter("ignore")
for fx in fixtures:
    for iid, est in fx["instances"].items():
        valid = []
        for r in fx["rows"]:
            if r["__iid"] != iid:
                continue
            vals = [float("nan") if r[n] is None else float(r[n]) for n in fx["names"]]
            try:
                y = est.transform([vals])[0]  # the twin, as served
            except ValueError:
                continue  # sklearn rejects NaN: the twin raises
            valid.append(vals)
            row_lanes.extend(float(v) for v in y)
        if valid:
            Y = est.transform(np.array(valid))
            batch_lanes.extend(float(v) for v in Y.ravel())
np.savez(out_path, row=np.array(row_lanes), batch=np.array(batch_lanes))
print(arch, "threads", nth, "lanes", len(row_lanes), len(batch_lanes))
