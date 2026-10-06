"""SplineTransformer(handle_missing='zeros'): sklearn then validates with
ensure_all_finite=False (_polynomial.py:1001), so the twin ANSWERS on
+-inf. Twin vs entry on +-inf, NaN, per extrapolation."""
import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import shim  # noqa: F401
import math, warnings
warnings.simplefilter("ignore")
import numpy as np, pyarrow as pa
from confit import DuckDBInferFn
from sklearn.preprocessing import SplineTransformer
from sql_transform._udf import PythonTransform
from sql_transform.native import to_native, NotNative
from sql_transform.native._registry import query
from sql_transform.native._check import _same

def serve(sql, rows, fn):
    try:
        f = DuckDBInferFn(sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[fn])
        return f.infer_arrow(rows).to_pylist()
    except Exception as e:
        return f"RAISES {str(e).splitlines()[0][:90]}"
X = np.random.default_rng(0).uniform(1, 10, size=(40, 1))
for ext in ["constant", "linear", "continue", "periodic", "error"]:
    for deg in (1, 3):
        est = SplineTransformer(degree=deg, n_knots=4, extrapolation=ext, handle_missing="zeros").fit(X)
        w = est.transform(X[:1]).shape[1]
        step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64())]), pa.struct([(f"f{j}", pa.float64()) for j in range(w)]))
        try:
            nat = to_native(step, strict=True)
        except NotNative as e:
            print(ext, deg, "NotNative", e); continue
        for v in (math.inf, -math.inf, math.nan, 1e308):
            rows = pa.table({"__iid": pa.array([0], pa.int64()), "x0": pa.array([v])})
            t = serve(query(step), rows, step); n = serve(query(step), rows, nat)
            if isinstance(t, list) and isinstance(n, list):
                ok = all(_same(t[0][k], n[0][k], 0) for k in t[0])
                print(f"{ext:<9} deg{deg} x={v}: {'EQUAL' if ok else 'DIFFER'}  twin {[round(z,4) if isinstance(z,float) else z for z in t[0].values()]}  entry {[round(z,4) if isinstance(z,float) else z for z in n[0].values()]}")
            else:
                print(f"{ext:<9} deg{deg} x={v}: twin {str(t)[:90]} | entry {str(n)[:90]}")
