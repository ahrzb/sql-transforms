"""Run the gate's own check over every fixture with +-inf added to EDGES
(seeds 0..N-1): does +-inf alone expose a parity breach? Then add a
handle_missing='zeros' periodic spline fixture."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/tolerated")
import shim  # noqa
import math, time, warnings, collections
from threadpoolctl import threadpool_limits
from sklearn.preprocessing import SplineTransformer
import sql_transform.native.catalog_test as ct
from sql_transform.native import NotNative, to_native
from sql_transform.native._check import check, ParityError
warnings.simplefilter("ignore")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ct.EDGES[:] = [*ct.EDGES, math.inf, -math.inf, math.inf, -math.inf]
import builtins
ct.round = lambda v: (2**62 if v == math.inf else -(2**62) if v == -math.inf else builtins.round(v))
extra = {"Spline zeros periodic d1 k4": lambda: SplineTransformer(degree=1, n_knots=4, extrapolation="periodic", handle_missing="zeros"),
         "Spline zeros periodic d3 k6": lambda: SplineTransformer(degree=3, n_knots=6, extrapolation="periodic", handle_missing="zeros"),
         "Spline zeros constant d3 k5": lambda: SplineTransformer(degree=3, n_knots=5, handle_missing="zeros")}
items = [(f"{c.__name__}[{j}]", f, j) for c, fs in ct.FIXTURES.items() for j, f in enumerate(fs)]
items += [(k, f, 99) for k, f in extra.items()]
t0 = time.time(); fails = []; n = 0; infrows = 0
with threadpool_limits(limits=1):
    for lab, make, j in items:
        for seed in range(N):
            step = ct._step(make, seed, j)
            try:
                nat = to_native(step, strict=True)
            except NotNative:
                continue
            rows = ct._rows(step, seed, getattr(make, "positive", False))
            infrows += sum(1 for r in rows.to_pylist() if any(isinstance(v, float) and math.isinf(v) for k, v in r.items() if k != "__iid"))
            try:
                n += check(step, nat, rows)
            except ParityError as e:
                fails.append((lab, seed, str(e)[:160]))
print(f"{time.time()-t0:.0f}s, compared {n}, rows holding +-inf {infrows}")
for f in fails: print("  BREACH", f)
