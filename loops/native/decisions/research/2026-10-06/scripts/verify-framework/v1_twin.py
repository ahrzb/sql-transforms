"""Independent: regenerate PCA(whiten) fixtures seeds 0..N-1, and dump the
twin's row-by-row lanes and the batched lanes under the current BLAS kernel."""
import math, pickle, sys, warnings, numpy as np, threadpoolctl
from sklearn.decomposition import PCA
from fixtures import _step, _rows
warnings.simplefilter("ignore")
N = int(sys.argv[1]); out = sys.argv[2]
recs = []  # (seed, iid, x, est)
ests = {}
for seed in range(N):
    step = _step(lambda: PCA(whiten=True), seed, 0)
    names = step.takes.names
    for r in _rows(step, seed).to_pylist():
        iid = r["__iid"]
        if iid is None: continue
        x = [math.nan if r[n] is None else float(r[n]) for n in names]
        if not all(math.isfinite(v) for v in x): continue
        est = step.instances[iid]
        ests[(seed, iid)] = est
        recs.append(((seed, iid), x))
row = []; Ms = []
for key, x in recs:
    est = ests[key]
    row.append(np.array(est.transform([x])[0], float))
    Ms.append((est.mean_.reshape(1, -1) @ est.components_.T)[0])
# batched per estimator, in fixture order
from collections import defaultdict
g = defaultdict(list)
for i, (key, x) in enumerate(recs): g[key].append(i)
bat = [None] * len(recs)
for key, idx in g.items():
    B = ests[key].transform(np.array([recs[i][1] for i in idx]))
    for i, b in zip(idx, B): bat[i] = np.array(b, float)
arch = threadpoolctl.threadpool_info()[0].get("architecture")
pickle.dump(dict(arch=arch, recs=recs, ests=ests, row=row, bat=bat, M=Ms), open(out, "wb"))
print(arch, len(recs), "rows", sum(len(r) for r in row), "lanes", len(g), "estimators")
