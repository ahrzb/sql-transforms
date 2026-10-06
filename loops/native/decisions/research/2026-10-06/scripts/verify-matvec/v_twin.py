"""Independent twin runner. Under the caller's OPENBLAS env, for every valid
(fixture, instance, row): the served twin est.transform([vals]) (row), the
batch transform per instance, and this process's projected mean in two
spellings: mean_.reshape(1,-1) @ C.T (sklearn's) and C @ mean_."""
import pickle, sys, warnings
import numpy as np
from threadpoolctl import threadpool_info

fx_path, out = sys.argv[1], sys.argv[2]
info = [d for d in threadpool_info() if d["internal_api"] == "openblas"][0]
fixtures = pickle.load(open(fx_path, "rb"))
warnings.simplefilter("ignore")
row, batch, mc1, mc2, key = [], [], [], [], []
for fi, fx in enumerate(fixtures):
    for iid, est in fx["instances"].items():
        C = est.components_
        m1 = (est.mean_.reshape(1, -1) @ C.T)[0]
        m2 = C @ est.mean_
        good = []
        for ri, r in enumerate(fx["rows"]):
            if r["__iid"] != iid:
                continue
            v = [float("nan") if r[nm] is None else float(r[nm]) for nm in fx["names"]]
            try:
                y = est.transform([v])[0]
            except ValueError:
                continue
            good.append(v)
            for k in range(C.shape[0]):
                row.append(float(y[k])); mc1.append(float(m1[k])); mc2.append(float(m2[k]))
                key.append((fx["seed"], iid, ri, k))
        if good:
            batch.extend(float(t) for t in est.transform(np.array(good)).ravel())
np.savez(out, row=np.array(row), batch=np.array(batch), mc1=np.array(mc1), mc2=np.array(mc2), key=np.array(key))
print(info["architecture"], info["num_threads"], info["version"], len(row), len(batch))
