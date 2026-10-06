"""E1 step 1: the catalog's own fixtures for PCA(whiten=True), seeds 0..N-1,
pickled as (seed, est, x) cases. Fit once here so every twin variant
transforms the same fitted estimators."""
import math, pickle, sys, warnings
from sklearn.decomposition import PCA
from fixtures import _step, _rows

N = int(sys.argv[1]); out = sys.argv[2]
cases = []
warnings.simplefilter("ignore")
for seed in range(N):
    step = _step(lambda: PCA(whiten=True), seed, 0)
    rows = _rows(step, seed).to_pylist()
    names = step.takes.names
    for r in rows:
        iid = r["__iid"]
        if iid is None:
            continue
        x = [math.nan if r[n] is None else float(r[n]) for n in names]
        if any(math.isnan(v) or math.isinf(v) for v in x):
            continue  # the twin raises (sklearn rejects NaN/inf)
        cases.append((seed, step.instances[iid], x))
pickle.dump(cases, open(out, "wb"))
print(len(cases), "rows", sum(c[1].components_.shape[0] for c in cases), "lanes")
