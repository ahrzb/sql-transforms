"""The repo's own gate (`native.check`) on the case spline_zeros.py found:
SplineTransformer(degree=1, n_knots=4, extrapolation='periodic',
handle_missing='zeros'), rows x = +inf, -inf (the twin answers there)."""
import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import shim  # noqa: F401
import math, warnings
warnings.simplefilter("ignore")
import numpy as np, pyarrow as pa
from sklearn.preprocessing import SplineTransformer
from sql_transform._udf import PythonTransform
from sql_transform.native import to_native, check, ParityError
X = np.random.default_rng(0).uniform(1, 10, size=(40, 1))
for nk in (2, 3, 4, 6):
    est = SplineTransformer(degree=1, n_knots=nk, extrapolation="periodic", handle_missing="zeros").fit(X)
    w = est.transform(X[:1]).shape[1]
    step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64())]), pa.struct([(f"f{j}", pa.float64()) for j in range(w)]))
    rows = pa.table({"__iid": pa.array([0, 0], pa.int64()), "x0": pa.array([math.inf, -math.inf])})
    try:
        print(nk, "compared", check(step, to_native(step, strict=True), rows))
    except ParityError as e:
        print(nk, "ParityError:", str(e)[:160])
