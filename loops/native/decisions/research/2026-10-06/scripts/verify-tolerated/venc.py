import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/tolerated")
import shim  # noqa
import math, warnings
import numpy as np, pyarrow as pa
from confit import DuckDBInferFn
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sql_transform._udf import PythonTransform
from sql_transform.native import to_native
from sql_transform.native._registry import query
warnings.simplefilter("ignore")
rng = np.random.default_rng(5)
X = np.empty((30, 2), dtype=object); X[:, 0] = rng.choice(["p", "q"], 30); X[:, 1] = [float(v) for v in rng.integers(0, 3, 30)]
for est in (OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(X),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-2).fit(X)):
    w = est.transform(X[:1]).shape[1]
    step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.string()), ("x1", pa.float64())]), pa.struct([(f"f{j}", pa.float64()) for j in range(w)]))
    nat = to_native(step, strict=True)
    for v in (math.inf, -math.inf, math.nan, 1.0):
        t = pa.table({"__iid": pa.array([0], pa.int64()), "x0": pa.array(["p"]), "x1": pa.array([v], pa.float64())})
        r = []
        for fn in (step, nat):
            try: r.append(DuckDBInferFn(query(step), row_tables={"__THIS__": t.schema}, static_tables={}, udfs=[fn]).infer_arrow(t).to_pylist()[0])
            except Exception as e: r.append("RAISE " + str(e).splitlines()[0][-60:])
        print(type(est).__name__, v, "twin", r[0], "entry", r[1], "same", r[0] == r[1])
