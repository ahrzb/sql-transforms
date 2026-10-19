"""What a generic validation guard costs a native entry, served by confit
(release build): the entry as is, against the same entry with lane 0
wrapped in `CASE WHEN <guard> THEN error('...') ELSE lane0 END`, where the
guard is, over the declared numeric features,

  inf:  abs(x_i) = inf                  (sklearn's ensure_all_finite="allow-nan")
  fin:  abs(x_i) = inf OR x_i = NaN     (ensure_all_finite=True)
  all:  the `fin` guard on every lane, not only lane 0

Timed: median of CALLS 64-row calls (as benchmarks/bench_native.py), and one
10,000-row call, rows every twin answers. Also: the build, and that the
guarded entry raises on an inf row where the unguarded one answers.
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
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, QuantileTransformer
from threadpoolctl import threadpool_limits

from sql_transform._udf import PythonTransform
from sql_transform.native import _registry, to_native
from sql_transform.native._helpers import f64, isnan
from sql_transform.native._registry import query

warnings.simplefilter("ignore")
CALLS = 60


def guarded(orig, mode):
    def tr(est, x, types):
        out = list(orig(est, x, types))
        conds = []
        for xi, t in zip(x, types, strict=True):
            if t == pa.string():
                continue
            c = S.fn("abs", xi) == f64(math.inf)
            if mode in ("fin", "all"):
                c = c | isnan(xi)
            conds.append(c)
        g = conds[0]
        for c in conds[1:]:
            g = g | c
        err = S.fn("error", S.lit(f"{type(est).__name__}: input contains infinity or NaN"))
        if mode == "all":
            return [S.case(g, err).otherwise(o) for o in out]
        out[0] = S.case(g, err).otherwise(out[0])
        return out

    return tr


def step_of(est_factory, k, rng, n_inst=1):
    inst = {}
    for i in range(n_inst):
        X = rng.normal(5, 2, size=(200, k))
        inst[i] = est_factory().fit(X)
    w = np.asarray(inst[0].transform(X[:1])).reshape(1, -1).shape[1]
    takes = pa.schema([(f"x{j}", pa.float64()) for j in range(k)])
    ret = pa.float64() if w == 1 else pa.struct([(f"f{j}", pa.float64()) for j in range(w)])
    return PythonTransform("tf", inst, takes, ret)


def rows(step, n, rng):
    cols = {"__iid": pa.array(rng.integers(0, len(step.instances), n), pa.int64())}
    for f in step.takes:
        cols[f.name] = pa.array(rng.normal(5, 2, n), pa.float64())
    return pa.table(cols)


def med_call(fn, t, calls=CALLS):
    fn.infer_arrow(t)
    ts = []
    for _ in range(calls):
        a = time.perf_counter()
        fn.infer_arrow(t)
        ts.append(time.perf_counter() - a)
    return statistics.median(ts)


def build(step, native, schema):
    a = time.perf_counter()
    fn = DuckDBInferFn(query(step), row_tables={"__THIS__": schema}, static_tables={}, udfs=[native])
    return fn, time.perf_counter() - a


def main():
    rng = np.random.default_rng(20261006)
    cases = [
        ("StandardScaler", StandardScaler),
        ("KBins(ordinal,5)", lambda: KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")),
        ("Quantile(100)", lambda: QuantileTransformer(n_quantiles=100)),
    ]
    print("| estimator | k | variant | build s | µs / 64-row call | µs / row @64 | µs / row @10k | Δ µs/row @10k |")
    print("|---|---|---|---|---|---|---|---|")
    for name, fac in cases:
        for k in (1, 4, 16, 32):
            if name.startswith("Quantile") and k > 32:
                continue
            step = step_of(fac, k, rng)
            t64 = rows(step, 64, rng)
            t10k = rows(step, 10_000, rng)
            schema = t64.schema
            cls = type(step.instances[0])
            orig = _registry._CATALOG[cls]
            base10k = None
            for mode in ("none", "inf", "fin", "all"):
                if mode != "none":
                    _registry._CATALOG[cls] = _registry.Entry(guarded(orig.translate, mode), orig.ulps, orig.per_estimator)
                try:
                    native = to_native(step, strict=True)
                    fn, b = build(step, native, schema)
                    c64 = med_call(fn, t64)
                    c10k = med_call(fn, t10k, 15)
                    # does the guard fire on an inf row, and only then?
                    bad = t64.set_column(1, "x0", pa.array([math.inf] + [5.0] * 63, pa.float64()))
                    try:
                        fn.infer_arrow(bad)
                        fired = "answers"
                    except Exception as e:  # noqa: BLE001
                        fired = "raises"
                finally:
                    _registry._CATALOG[cls] = orig
                pr = c10k / 10_000 * 1e6
                if base10k is None:
                    base10k = pr
                print(f"| {name} | {k} | {mode} ({fired} on inf) | {b:.3f} | {c64*1e6:.1f} | {c64/64*1e6:.3f} | {pr:.4f} | {pr-base10k:+.4f} |", flush=True)
            if k == 4:
                twin_fn = DuckDBInferFn(query(step), row_tables={"__THIS__": schema}, static_tables={}, udfs=[step])
                c = med_call(twin_fn, t64, 10)
                print(f"| {name} | {k} | twin (PythonTransform) | - | {c*1e6:.0f} | {c/64*1e6:.1f} | - | - |", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
