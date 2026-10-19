"""Spot checks of the report's named examples, with my own fits."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/tolerated")
import shim  # noqa: F401
import math, warnings
import numpy as np
import pyarrow as pa
from confit import DuckDBInferFn
import sklearn
from sklearn.preprocessing import KBinsDiscretizer, Binarizer, SplineTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import MissingIndicator
from sklearn.feature_selection import SelectKBest, f_classif
from sql_transform._udf import PythonTransform
from sql_transform.native import to_native
from sql_transform.native._registry import query
from sql_transform.native._check import check, ParityError

warnings.simplefilter("ignore")


def mk(est, d, w):
    ret = pa.float64() if w == 1 else pa.struct([(f"f{j}", pa.float64()) for j in range(w)])
    return PythonTransform("tf", {0: est}, pa.schema([(f"x{j}", pa.float64()) for j in range(d)]), ret)


def serve(step, fn, cols):
    n = len(cols[0])
    t = pa.table({"__iid": pa.array([0] * n, pa.int64()), **{f"x{j}": pa.array(c, pa.float64()) for j, c in enumerate(cols)}})
    try:
        return DuckDBInferFn(query(step, "__iid"), row_tables={"__THIS__": t.schema}, static_tables={}, udfs=[fn]).infer_arrow(t).to_pylist()
    except Exception as e:  # noqa: BLE001
        return "RAISE: " + str(e).splitlines()[0][:120]


def both(name, est, d, w, row):
    st = mk(est, d, w)
    nat = to_native(st, strict=True)
    cols = [[v] for v in row]
    tw = serve(st, st, cols)
    en = serve(st, nat, cols)
    vv = [math.nan if v is None else v for v in row]
    try:
        with sklearn.config_context(assume_finite=True), np.errstate(all="ignore"):
            ref = np.asarray(est.transform([vv] if d > 1 or not isinstance(est, IsotonicRegression) else vv)).reshape(-1).tolist()
    except Exception as e:  # noqa: BLE001
        ref = "RAISE " + str(e)[:60]
    print(f"{name:<44} row={row}\n    twin={tw}\n    entry={en}\n    sklearn(assume_finite)={ref}")


rng = np.random.default_rng(99)
X = rng.normal(0, 2, size=(300, 2))
kb = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile").fit(X)
both("KBins quantile ordinal 5, NULL", kb, 2, 2, [None, 0.1])
both("KBins quantile ordinal 5, -inf", kb, 2, 2, [-math.inf, 0.1])
kb2 = KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="uniform").fit(X)
both("KBins uniform onehot 4, NaN", kb2, 2, 8, [math.nan, 0.1])

b = Binarizer(threshold=0.5).fit(X)
both("Binarizer(0.5), NULL", b, 2, 2, [None, 1.0])
both("Binarizer(0.5), +inf/-inf", b, 2, 2, [math.inf, -math.inf])

Xs = rng.uniform(0, 10, size=(50, 1))
sp = SplineTransformer(n_knots=5, degree=3, handle_missing="error").fit(Xs)
w = sp.transform(Xs[:1]).shape[1]
both("Spline(error missing), NULL", sp, 1, w, [None])

xi = np.sort(rng.uniform(0, 10, 40)); yi = xi + rng.normal(0, 1, 40)
iso = IsotonicRegression(out_of_bounds="clip").fit(xi, yi)
print("iso fp ends", iso.f_(np.array([xi.min(), xi.max()])) if hasattr(iso, "f_") else None, iso.y_thresholds_[[0, -1]])
both("Isotonic(clip), NaN", iso, 1, 1, [math.nan])
both("Isotonic(clip), -inf", iso, 1, 1, [-math.inf])

Xm = rng.integers(0, 4, size=(50, 2)).astype(float); Xm[::7, 0] = -1
mi = MissingIndicator(missing_values=-1).fit(Xm)
wm = mi.transform(Xm[:1]).shape[1]
both("MissingIndicator(-1), NaN in x0", mi, 2, wm, [math.nan, 1.0])

yk = (rng.random(300) < 0.5).astype(int)
Xk = np.column_stack([X, yk + rng.normal(0, 0.1, 300)])
sk = SelectKBest(f_classif, k=1).fit(Xk, yk)
print("SelectKBest support", sk.get_support())
both("SelectKBest k=1, NaN in dropped x0", sk, 3, 1, [math.nan, 0.0, 1.0])
both("SelectKBest k=1, inf in dropped x0", sk, 3, 1, [math.inf, 0.0, 1.0])

# Batch: 1,000 rows, one NULL at 517
Xb = rng.normal(0, 1, size=(1000, 2)); cols = [Xb[:, 0].tolist(), Xb[:, 1].tolist()]
cols[0][517] = None
st = mk(kb, 2, 2); nat = to_native(st, strict=True)
tw = serve(st, st, cols); en = serve(st, nat, cols)
print("batch twin:", tw if isinstance(tw, str) else len(tw))
print("batch entry:", (len(en), en[517]) if not isinstance(en, str) else en, "bin_edges f0", kb.bin_edges_[0])
