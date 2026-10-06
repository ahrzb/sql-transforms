import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/tolerated")
import shim  # noqa
import math, warnings
import numpy as np, pyarrow as pa
from confit import DuckDBInferFn
from scipy.special import boxcox
from sklearn.preprocessing import PowerTransformer
from sql_transform._udf import PythonTransform
from sql_transform.native import to_native
from sql_transform.native._registry import query
warnings.simplefilter("ignore")
lams = [-1e-20, -1e-19, -3e-12, -50.0, 1e-19, 2.0]
X = np.random.default_rng(1).uniform(1, 5, size=(30, len(lams)))
est = PowerTransformer("box-cox", standardize=False).fit(X); est.lambdas_ = np.array(lams)
step = PythonTransform("tf", {0: est}, pa.schema([(f"x{j}", pa.float64()) for j in range(len(lams))]),
                       pa.struct([(f"f{j}", pa.float64()) for j in range(len(lams))]))
nat = to_native(step, strict=True)
for v in (math.inf, 0.0, -1.0, math.nan):
    t = pa.table({"__iid": pa.array([0], pa.int64()), **{f"x{j}": pa.array([v], pa.float64()) for j in range(len(lams))}})
    out = []
    for fn in (step, nat):
        try:
            out.append(DuckDBInferFn(query(step), row_tables={"__THIS__": t.schema}, static_tables={}, udfs=[fn]).infer_arrow(t).to_pylist()[0])
        except Exception as e:
            out.append("RAISE " + str(e).splitlines()[0][-60:])
    with np.errstate(all="ignore"):
        ref = [float(boxcox(v, l)) for l in lams]
    print(v, "twin:", out[0], "\n   entry:", out[1], "\n   scipy:", ref)
