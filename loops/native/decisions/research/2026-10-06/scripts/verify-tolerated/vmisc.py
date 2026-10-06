import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/tolerated")
import shim  # noqa
import math, warnings
import numpy as np, pyarrow as pa
from confit import DuckDBInferFn
from confit import sql as S
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sql_transform._udf import PythonTransform
from sql_transform.native import to_native, _registry
from sql_transform.native._registry import query
warnings.simplefilter("ignore")

def serve(step, fn, t):
    try:
        return DuckDBInferFn(query(step), row_tables={"__THIS__": t.schema}, static_tables={}, udfs=[fn]).infer_arrow(t).to_pylist()
    except Exception as e:
        return "RAISE " + str(e).splitlines()[0][-90:]

# 1. spline degree 0, constant, above the knots: a twin raise that is not validation
X = np.linspace(0, 10, 30)[:, None]
for nk in (2, 3, 4):
    est = SplineTransformer(degree=0, n_knots=nk, extrapolation="constant").fit(X)
    w = est.transform(X[:1]).shape[1]
    step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64())]), pa.float64() if w == 1 else pa.struct([(f"f{j}", pa.float64()) for j in range(w)]))
    nat = to_native(step, strict=True)
    t = pa.table({"__iid": pa.array([0, 0], pa.int64()), "x0": pa.array([11.0, -1.0], pa.float64())})
    print("spline deg0 constant n_knots", nk, "x=11,-1: twin", serve(step, step, t.slice(0,1)), "| entry", serve(step, nat, t.slice(0,1)), "| x=-1 twin", serve(step, step, t.slice(1,1)))

# 2. a lane-0 guard and a NULL instance id: the twin answers NULL without validating
Xs = np.random.default_rng(0).normal(size=(50, 3))
est = StandardScaler().fit(Xs)
step = PythonTransform("tf", {0: est}, pa.schema([(f"x{j}", pa.float64()) for j in range(3)]), pa.struct([(f"f{j}", pa.float64()) for j in range(3)]))
orig = _registry._CATALOG[StandardScaler]
def tr(e, x, types):
    out = list(orig.translate(e, x, types))
    g = None
    for xi in x:
        p = S.fn("abs", xi) == S.lit(math.inf)
        g = p if g is None else g | p
    out[0] = S.case(g, S.fn("error", S.lit("tf: inf"))).otherwise(out[0])
    return out
_registry._CATALOG[StandardScaler] = _registry.Entry(tr, 0)
try:
    nat = to_native(step, strict=True)
finally:
    _registry._CATALOG[StandardScaler] = orig
t = pa.table({"__iid": pa.array([None, 0], pa.int64()), "x0": pa.array([math.inf, 1.0]), "x1": pa.array([1.0, 1.0]), "x2": pa.array([1.0, 1.0])})
print("NULL id + inf: twin", serve(step, step, t), "| guarded entry", serve(step, nat, t))
