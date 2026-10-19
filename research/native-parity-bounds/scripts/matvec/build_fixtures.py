"""Build the catalog's PCA(whiten=True) fixtures (seeds 0..N-1, variant 0) ONCE,
under the default BLAS, and pickle the fitted instances and rows, so every
twin configuration transforms the same fitted state (fit itself is
BLAS/LAPACK-dependent, and serving never refits)."""

import pickle
import sys
import warnings

from sklearn.decomposition import PCA

sys.path.insert(0, __import__("os").path.dirname(__file__))
from repo_fixtures import _rows, _step  # noqa: E402

N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
out = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for seed in range(N):
        step = _step(lambda: PCA(whiten=True), seed)
        rows = _rows(step, seed).to_pylist()
        names = step.takes.names
        types = [str(f.type) for f in step.takes]
        out.append(
            dict(seed=seed, instances=step.instances, names=names, types=types, rows=rows)
        )
with open(sys.argv[1], "wb") as f:
    pickle.dump(out, f)
print("fixtures", len(out), "instances", sum(len(o["instances"]) for o in out))
