"""Guard spellings, timed interleaved (7 rounds x 5 calls of 10,000 rows,
median), on StandardScaler and KBinsDiscretizer(ordinal, 5 bins) over 32
features, one instance, confit release build:

  none        the entry as is
  inf_abs     lane 0: CASE WHEN abs(x_i) = inf OR ... THEN error()
  fin_abs     ... abs(x_i) = inf OR x_i = NaN ...
  fin_range   ... x_i > DBL_MAX OR x_i < -DBL_MAX ...  (NaN orders above +inf)
  nan_out     every lane j: CASE WHEN <fin_range on the features lane j
              reads> THEN NaN ELSE lane_j  ("non-finite in -> non-finite out")

and which inputs each spelling traps, in confit and in DuckDB.
"""

from __future__ import annotations

import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import shim  # noqa: F401

import math
import statistics
import time
import warnings

import numpy as np
import pyarrow as pa
from confit import DuckDBInferFn
from confit import sql as S
from confit.oracle import Oracle
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from threadpoolctl import threadpool_limits

from sql_transform._udf import PythonTransform
from sql_transform.native import _registry, to_native
from sql_transform.native._helpers import f64, isnan
from sql_transform.native._registry import query

warnings.simplefilter("ignore")
MAX = 1.7976931348623157e308


def pred(xi, how):
    if how == "inf_abs":
        return S.fn("abs", xi) == f64(math.inf)
    if how == "fin_abs":
        return (S.fn("abs", xi) == f64(math.inf)) | isnan(xi)
    if how in ("fin_range", "nan_out"):
        return (xi > f64(MAX)) | (xi < f64(-MAX))
    raise ValueError(how)


def variant(orig, how):
    def tr(est, x, types):
        out = list(orig(est, x, types))
        if how == "none":
            return out
        if how == "nan_out":
            # elementwise estimators: lane j reads feature j
            return [S.case(pred(x[j], how), f64(math.nan)).otherwise(o) for j, o in enumerate(out)]
        g = pred(x[0], how)
        for xi in x[1:]:
            g = g | pred(xi, how)
        out[0] = S.case(g, S.fn("error", S.lit("tf: non-finite input"))).otherwise(out[0])
        return out

    return tr


VARIANTS = ["none", "inf_abs", "fin_abs", "fin_range", "nan_out"]


def main():
    rng = np.random.default_rng(7)
    k = 32
    X = rng.normal(5, 2, size=(200, k))
    for name, est in [
        ("StandardScaler", StandardScaler().fit(X)),
        ("KBins(ordinal,5)", KBinsDiscretizer(n_bins=5, encode="ordinal").fit(X)),
    ]:
        step = PythonTransform(
            "tf", {0: est}, pa.schema([(f"x{j}", pa.float64()) for j in range(k)]),
            pa.struct([(f"f{j}", pa.float64()) for j in range(k)]),
        )
        rows = pa.table({"__iid": pa.array([0] * 10_000, pa.int64()),
                         **{f"x{j}": pa.array(rng.normal(5, 2, 10_000)) for j in range(k)}})
        cls = type(est)
        orig = _registry._CATALOG[cls]
        fns = {}
        for v in VARIANTS:
            _registry._CATALOG[cls] = _registry.Entry(variant(orig.translate, v), 0)
            try:
                nat = to_native(step, strict=True)
            finally:
                _registry._CATALOG[cls] = orig
            fns[v] = (nat, DuckDBInferFn(query(step), row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[nat]))
            fns[v][1].infer_arrow(rows)
        times = {v: [] for v in VARIANTS}
        for _ in range(7):
            for v in VARIANTS:
                f = fns[v][1]
                for _ in range(5):
                    a = time.perf_counter()
                    f.infer_arrow(rows)
                    times[v].append(time.perf_counter() - a)
        base = statistics.median(times["none"]) / 1e4 * 1e6
        print(f"{name}, {k} features, 10,000-row calls: µs/row (Δ vs none)")
        for v in VARIANTS:
            m = statistics.median(times[v]) / 1e4 * 1e6
            q1, q3 = np.percentile(np.array(times[v]) / 1e4 * 1e6, [25, 75])
            print(f"  {v:<10} {m:.3f}  (IQR {q1:.3f}-{q3:.3f})  Δ {m - base:+.3f} = {1e3 * (m - base) / k:+.1f} ns/feature")
        # which inputs trap
        probes = [math.inf, -math.inf, math.nan, None, MAX, -MAX, 5e-324, -0.0]
        for v in VARIANTS[1:]:
            nat, f = fns[v]
            res_c, res_d = [], []
            for p in probes:
                t = rows.slice(0, 1).set_column(1, "x0", pa.array([p], pa.float64()))
                try:
                    o = f.infer_arrow(t).to_pylist()[0]["f0"]
                    res_c.append(f"{o!r}")
                except Exception:  # noqa: BLE001
                    res_c.append("ERR")
                with Oracle() as orc:
                    nat.register(orc)
                    orc.load("__THIS__", t)
                    d = orc.try_answer(query(step))
                res_d.append("ERR" if not isinstance(d, pa.Table) else repr(d.to_pylist()[0]["f0"]))
            same = res_c == res_d
            print(f"  traps {v:<10} on {probes}: confit {res_c} duckdb-same={same}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
