"""Edge experiments beside probe.py:

A. Encoders on a row with a string feature: sklearn validates an object
   array, whose finiteness check skips infinity. Does the twin answer +inf?
B. Box-Cox with a negative lambda at +inf: the entry's answer.
C. Batch semantics: one bad row in a 1,000-row call.
D. A lane-0 guard (and the unknown-id trap, which is built the same way)
   when the serving query reads a single other field: does it fire, in
   confit and in DuckDB?
"""

from __future__ import annotations

import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import shim  # noqa: F401

import math
import warnings

import numpy as np
import pyarrow as pa
from confit import DuckDBInferFn
from confit import sql as S
from confit.oracle import Oracle
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
    TargetEncoder,
)

from sql_transform._udf import PythonTransform
from sql_transform.native import _registry, to_native
from sql_transform.native._helpers import f64
from sql_transform.native._registry import query

warnings.simplefilter("ignore")


def serve(sql, rows, fn):
    try:
        f = DuckDBInferFn(sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[fn])
        return f.infer_arrow(rows).to_pylist()
    except Exception as e:  # noqa: BLE001
        return f"RAISES: {str(e).splitlines()[0][:110]}"


def struct(w):
    return pa.float64() if w == 1 else pa.struct([(f"f{j}", pa.float64()) for j in range(w)])


print("== A. encoders, row = (string, number)")
rng = np.random.default_rng(0)
X = np.empty((40, 2), dtype=object)
X[:, 0] = rng.choice(["a", "b", "c"], 40)
X[:, 1] = [float(v) for v in rng.integers(1, 4, 40)]
y = (rng.random(40) < 0.5).astype(int)
for est in [
    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    TargetEncoder(),
    OneHotEncoder(sparse_output=False),
]:
    est.fit(X, y)
    w = np.asarray(est.transform(X[:1])).shape[1]
    step = PythonTransform("tf", {0: est}, pa.schema([("s", pa.string()), ("x", pa.float64())]), struct(w))
    nat = to_native(step, strict=True)
    for v in (math.inf, -math.inf, math.nan, 2.0):
        rows = pa.table({"__iid": pa.array([0], pa.int64()), "s": ["a"], "x": pa.array([v], pa.float64())})
        print(f"  {type(est).__name__}({getattr(est, 'handle_unknown', '')}) x={v}: twin {serve(query(step), rows, step)}  entry {serve(query(step), rows, nat)}")
    # the same estimator fitted on numbers only: the float path
X2 = rng.integers(1, 4, size=(40, 2)).astype(float)
for est in [OneHotEncoder(sparse_output=False, handle_unknown="ignore")]:
    est.fit(X2)
    w = np.asarray(est.transform(X2[:1])).shape[1]
    step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64()), ("x1", pa.float64())]), struct(w))
    nat = to_native(step, strict=True)
    rows = pa.table({"__iid": pa.array([0], pa.int64()), "x0": pa.array([math.inf]), "x1": pa.array([2.0])})
    print(f"  numeric-only OneHotEncoder(ignore) x0=inf: twin {serve(query(step), rows, step)}  entry {serve(query(step), rows, nat)}")

print("== B. Box-Cox, pinned lambdas, x = +inf")
Xp = rng.uniform(1, 10, size=(40, 4))
pt = PowerTransformer("box-cox", standardize=False).fit(Xp)
pt.lambdas_ = np.array([-50.0, -1.0, -1e-3, 0.5])
step = PythonTransform("tf", {0: pt}, pa.schema([(f"x{j}", pa.float64()) for j in range(4)]), struct(4))
nat = to_native(step, strict=True)
rows = pa.table({"__iid": pa.array([0], pa.int64()), **{f"x{j}": pa.array([math.inf]) for j in range(4)}})
print("  lambdas", pt.lambdas_.tolist())
print("  twin ", serve(query(step), rows, step))
print("  entry", serve(query(step), rows, nat))
from scipy.special import boxcox
print("  scipy boxcox(inf, lam)", [float(boxcox(math.inf, l)) for l in pt.lambdas_], " -1/lam", [-1 / l for l in pt.lambdas_])
rows0 = pa.table({"__iid": pa.array([0], pa.int64()), **{f"x{j}": pa.array([1e308]) for j in range(4)}})
print("  x=1e308: twin", serve(query(step), rows0, step), "\n           entry", serve(query(step), rows0, nat))

print("== C. one NULL among 1,000 rows, KBinsDiscretizer(ordinal) over 2 features")
Xk = rng.normal(0, 1, size=(200, 2))
kb = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile").fit(Xk)
step = PythonTransform("tf", {0: kb}, pa.schema([("x0", pa.float64()), ("x1", pa.float64())]), struct(2))
nat = to_native(step, strict=True)
x0 = rng.normal(0, 1, 1000).tolist()
x0[517] = None
rows = pa.table({"__iid": pa.array([0] * 1000, pa.int64()), "x0": pa.array(x0, pa.float64()), "x1": pa.array(rng.normal(0, 1, 1000))})
t = serve(query(step), rows, step)
n = serve(query(step), rows, nat)
print("  twin :", t if isinstance(t, str) else f"{len(t)} rows")
print("  entry:", n if isinstance(n, str) else f"{len(n)} rows; row 517 = {n[517]} (bins 0..4; edges {kb.bin_edges_[0].round(3).tolist()})")

print("== D. a lane-0 trap, reading one other field")
Xs = rng.normal(5, 2, size=(50, 4))
ss = StandardScaler().fit(Xs)
step = PythonTransform("tf", {0: ss}, pa.schema([(f"x{j}", pa.float64()) for j in range(4)]), struct(4))
orig = _registry._CATALOG[StandardScaler]


def guarded(est, x, types):
    out = list(orig.translate(est, x, types))
    g = S.fn("abs", x[0]) == f64(math.inf)
    for xi in x[1:]:
        g = g | (S.fn("abs", xi) == f64(math.inf))
    out[0] = S.case(g, S.fn("error", S.lit("tf: input contains infinity"))).otherwise(out[0])
    return out


_registry._CATALOG[StandardScaler] = _registry.Entry(guarded, 0)
try:
    nat = to_native(step, strict=True)
finally:
    _registry._CATALOG[StandardScaler] = orig
args = "__iid, x0, x1, x2, x3"
for label, iid, x3 in [("inf in x3", 0, math.inf), ("unknown id 99", 99, 1.0)]:
    rows = pa.table({"__iid": pa.array([iid], pa.int64()), **{f"x{j}": pa.array([1.0 if j < 3 else x3]) for j in range(4)}})
    for q in (f"SELECT tf({args}).f3 AS o FROM __THIS__", f"SELECT tf({args}).f0 AS o FROM __THIS__"):
        c = serve(q, rows, nat)
        with Oracle() as o:
            nat.register(o)
            o.load("__THIS__", rows)
            d = o.try_answer(q)
        d = d.to_pylist() if isinstance(d, pa.Table) else f"RAISES: {str(d).splitlines()[0][:80]}"
        tw = serve(q, rows, step)
        print(f"  {label}; {q.split(' AS')[0][7:]}: confit {c} | duckdb {d} | twin {str(tw)[:60]}")
