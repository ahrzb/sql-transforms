"""SplineTransformer(extrapolation='periodic', handle_missing='zeros') at +-inf:
does the twin answer, and does native.check raise ParityError?"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/tolerated")
import shim  # noqa: F401
import math, warnings
import numpy as np
import pyarrow as pa
from sklearn.preprocessing import SplineTransformer
from sql_transform._udf import PythonTransform
from sql_transform.native import to_native
from sql_transform.native._check import check, ParityError

warnings.simplefilter("ignore")
rng = np.random.default_rng(3)
X = rng.uniform(-4, 9, size=(80, 2))
for degree in (1, 2, 3):
    for nk in (2, 3, 4, 5, 6, 9):
        for hm in ("zeros", "error"):
            try:
                est = SplineTransformer(n_knots=nk, degree=degree, extrapolation="periodic", handle_missing=hm).fit(X)
            except Exception as e:  # noqa: BLE001
                print(degree, nk, hm, "fit fails", str(e)[:60]); continue
            w = est.transform(X[:1]).shape[1]
            step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64()), ("x1", pa.float64())]),
                                   pa.struct([(f"f{j}", pa.float64()) for j in range(w)]))
            try:
                nat = to_native(step, strict=True)
            except Exception as e:  # noqa: BLE001
                print(degree, nk, hm, "NotNative", str(e)[:80]); continue
            rows = pa.table({"__iid": pa.array([0] * 5, pa.int64()),
                             "x0": pa.array([math.inf, -math.inf, math.nan, None, 1.0], pa.float64()),
                             "x1": pa.array([1.0, 2.0, 3.0, 4.0, 5.0], pa.float64())})
            try:
                n = check(step, nat, rows)
                print(f"deg {degree} n_knots {nk} {hm}: ok, compared {n}")
            except ParityError as e:
                print(f"deg {degree} n_knots {nk} {hm}: ParityError {str(e)[:150]}")
            with np.errstate(all="ignore"):
                try:
                    print("     twin(+inf):", np.round(est.transform([[math.inf, 1.0]])[0], 3).tolist())
                except Exception as e:  # noqa: BLE001
                    print("     twin(+inf) raises", str(e)[:50])
