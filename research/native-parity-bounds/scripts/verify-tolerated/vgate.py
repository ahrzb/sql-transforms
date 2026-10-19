"""Gate rows the twin raises on, counted directly with est.transform (no
confit, no check()): seeds 0..N-1. Also: how many gate rows hold +-inf, and
how many twin-raise rows the entry answers with a fully finite row."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/tolerated")
import shim  # noqa: F401
import collections, math, time, warnings
import numpy as np
from threadpoolctl import threadpool_limits
from sql_transform.native import NotNative, to_native
from sql_transform.native.catalog_test import FIXTURES, _rows, _step
from sql_transform._udf import _as_feature

warnings.simplefilter("ignore")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
tot = collections.Counter(); rz = collections.Counter(); infrows = 0; reasons = collections.Counter()
t0 = time.time()
with threadpool_limits(limits=1):
    for cls, fs in FIXTURES.items():
        for j, make in enumerate(fs):
            for seed in range(N):
                step = _step(make, seed, j)
                try:
                    to_native(step, strict=True)
                except NotNative:
                    continue
                rows = _rows(step, seed, getattr(make, "positive", False)).to_pylist()
                tys = step.take_types
                for r in rows:
                    tot[cls.__name__] += 1
                    vals = [r[f.name] for f in step.takes]
                    if any(isinstance(v, float) and math.isinf(v) for v in vals):
                        infrows += 1
                    if r["__iid"] is None:
                        continue
                    est = step.instances[r["__iid"]]
                    x = [_as_feature(v, t) for v, t in zip(vals, tys)]
                    try:
                        with np.errstate(all="ignore"):
                            est.transform([x])
                    except Exception as e:  # noqa: BLE001
                        rz[cls.__name__] += 1
                        m = str(e).split("\n")[0]
                        key = ("NaN" if "NaN" in m else "inf" if "infinity" in m else
                               "unknown cat" if "unknown categor" in m else
                               "positive" if "strictly positive" in m else
                               "bounds" if "bounds" in m or "range" in m else m[:50])
                        reasons[key] += 1
print(f"seeds 0..{N-1}: {time.time()-t0:.0f}s; rows {sum(tot.values())}, twin raises {sum(rz.values())} = {sum(rz.values())/sum(tot.values()):.1%}; rows holding +-inf: {infrows}")
for k in tot:
    if rz[k]:
        print(f"  {k:<26} {rz[k]:>4}/{tot[k]:<5} {rz[k]/tot[k]:.1%}")
print("reasons:", dict(reasons))
