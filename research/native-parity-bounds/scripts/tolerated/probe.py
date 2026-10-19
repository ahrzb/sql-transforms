"""For every catalog configuration (catalog_test.FIXTURES), fit on a small
controlled matrix, then serve probe rows (one feature set to a probe value,
the others at a typical fitted value) through confit with the twin
(PythonTransform) and with the native entry, one row at a time.

Classifies each (configuration, feature, probe) where the twin raises:
  c   the entry raises too
  a   every lane that reads the offending feature is non-finite or NULL
  b   every such lane is finite (silently plausible)
  ab  some finite, some not
  b0  no lane reads the feature (a selector dropped it), all lanes finite
and where the twin answers: '=' (within bound) or 'DIFF'.

Also evaluates three generic guards against the twin's raise set.
Writes probe.json next to this script.
"""

from __future__ import annotations

import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import shim  # noqa: F401

import json
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
from confit import DuckDBInferFn
from threadpoolctl import threadpool_limits

from sql_transform._udf import PythonTransform
from sql_transform.native import NotNative, bound, to_native
from sql_transform.native._check import _same
from sql_transform.native._registry import query
from sql_transform.native.catalog_test import FIXTURES, _runs

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

PROBES = [
    ("+inf", math.inf),
    ("-inf", -math.inf),
    ("NaN", math.nan),
    ("NULL", None),
    ("1e308", 1e308),
    ("-1e308", -1e308),
    ("0", 0.0),
    ("-1", -1.0),
    ("1e6", 1e6),
    ("-1e6", -1e6),
    ("0.5", 0.5),
]
# finite values used only to find which lanes read a feature
SPREAD = [-50.0, -0.5, 0.5, 1.5, 2.0, 3.0, 4.5, 7.0, 9.5, 50.0]


def fit_data(factory, rng):
    proto_runs = _runs(factory())
    proto = proto_runs[0]
    tags = proto.__sklearn_tags__()
    one_d = not tags.input_tags.two_d_array
    categorical = tags.input_tags.categorical
    regression = proto_runs[-1].__sklearn_tags__().estimator_type == "regressor"
    d = 1 if one_d else 3
    n = 40
    if categorical:
        X = rng.integers(1, 4, size=(n, d)).astype(float)
    else:
        X = np.round(rng.uniform(1, 10, size=(n, d)), 2)
    if hasattr(proto, "missing_values"):
        mv = float(proto.missing_values)
        # missing in feature 0 only at fit: a missing feature 1 or 2 is new
        X[rng.random(n) < 0.2, 0] = mv
        X[0, 0] = mv
    y = (rng.random(n) < 0.5).astype(int)
    y[:4] = (0, 1, 0, 1)
    if regression or one_d:
        y = np.nan_to_num(X[:, 0]) + rng.normal(0, 1, n)
    return X, y, d


def build(factory, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = fit_data(factory, rng)
    est = factory().fit(X, y)
    width = np.asarray(est.transform(X[:1])).reshape(1, -1).shape[1]
    takes = pa.schema([(f"x{i}", pa.float64()) for i in range(d)])
    returns = (
        pa.float64()
        if width == 1
        else pa.struct([(f"f{j}", pa.float64()) for j in range(width)])
    )
    step = PythonTransform("tf", {0: est}, takes, returns)
    base = [float(np.nanmedian(X[:, j])) if not np.isnan(X[:, j]).all() else 1.0 for j in range(d)]
    if hasattr(_runs(est)[0], "missing_values"):
        mv = float(_runs(est)[0].missing_values)
        base = [b if b != mv else b + 1 for b in base]
    return step, est, X, base, d


def table(rows, d):
    cols = {"__iid": pa.array([0] * len(rows), pa.int64())}
    for j in range(d):
        cols[f"x{j}"] = pa.array([r[j] for r in rows], pa.float64())
    return pa.table(cols)


class Server:
    def __init__(self, step, fn):
        self.sql = query(step, "__iid")
        sch = pa.schema([pa.field("__iid", pa.int64()), *step.takes])
        self.f = DuckDBInferFn(
            self.sql, row_tables={"__THIS__": sch}, static_tables={}, udfs=[fn]
        )

    def one(self, t):
        try:
            return self.f.infer_arrow(t).to_pylist()[0]
        except Exception as e:  # noqa: BLE001
            return e

    def many(self, t):
        try:
            return self.f.infer_arrow(t).to_pylist()
        except Exception:
            return [self.one(t.slice(i, 1)) for i in range(t.num_rows)]


def lanes(ans):
    return list(ans.values())


def finite(v):
    return v is not None and isinstance(v, float) and math.isfinite(v)


def guard_tags(est):
    t = est.__sklearn_tags__()
    return t.input_tags.allow_nan, getattr(t, "no_validation", False)


def main():
    out = []
    t0 = time.time()
    for cls, factories in FIXTURES.items():
        for j, factory in enumerate(factories):
            label = f"{cls.__name__}[{j}]"
            try:
                step, est, X, base, d = build(factory)
            except Exception as e:  # noqa: BLE001
                out.append({"cfg": label, "error": f"fit: {type(e).__name__}: {e}"[:200]})
                continue
            try:
                native = to_native(step, strict=True)
            except NotNative as e:
                out.append({"cfg": label, "error": f"NotNative: {e}"[:200]})
                continue
            try:
                ulps = bound(step)
                tw, nv = Server(step, step), Server(step, native)
            except Exception as e:  # noqa: BLE001
                out.append({"cfg": label, "error": f"serve: {e}"[:200]})
                continue
            allow_nan, no_val = guard_tags(est)
            rec = {
                "cfg": label,
                "cls": cls.__name__,
                "repr": repr(est)[:160],
                "allow_nan": allow_nan,
                "no_validation": no_val,
                "ulps": ulps,
                "rows": [],
            }
            for p in range(d):
                # lanes that read feature p
                fin_rows = []
                for v in SPREAD + [float(x) for x in np.unique(X[:, p][~np.isnan(X[:, p])])[:6]]:
                    r = list(base)
                    r[p] = v
                    fin_rows.append(r)
                ft = tw.many(table(fin_rows, d))
                answered = [a for a in ft if isinstance(a, dict)]
                dep = set()
                if answered:
                    keys = list(answered[0].keys())
                    for k in keys:
                        vals = {repr(a[k]) for a in answered}
                        if len(vals) > 1:
                            dep.add(k)
                prow = []
                for name, v in PROBES:
                    r = list(base)
                    r[p] = v
                    prow.append(r)
                pt = table(prow, d)
                twin = tw.many(pt)
                ent = nv.many(pt)
                for (name, v), a, b in zip(PROBES, twin, ent, strict=True):
                    row = {"feat": p, "probe": name}
                    if isinstance(a, Exception):
                        row["twin"] = "raise: " + str(a).split("\n")[0][-160:]
                        if isinstance(b, Exception):
                            row["entry"] = "raise: " + str(b)[:120]
                            row["cls"] = "c"
                        else:
                            row["entry"] = b
                            # the twin with sklearn's finiteness check off
                            import sklearn

                            vals = [math.nan if q is None else q for q in prow[PROBES.index((name, v))]]
                            try:
                                with sklearn.config_context(assume_finite=True), np.errstate(all="ignore"):
                                    ref = [float(z) for z in np.asarray(est.transform([vals]), dtype=float).reshape(-1)]
                                row["ref"] = ref
                                row["ref_same"] = all(
                                    _same(rv, ev, ulps) for rv, ev in zip(ref, b.values(), strict=True)
                                )
                            except Exception as e:  # noqa: BLE001
                                row["ref"] = "raise: " + str(e).split("\n")[0][-120:]
                                row["ref_same"] = None
                            dl = [b[k] for k in b if k in dep]
                            if not dep:
                                row["cls"] = "b0" if all(finite(x) for x in lanes(b)) else "a0"
                            elif all(finite(x) for x in dl):
                                row["cls"] = "b"
                            elif not any(finite(x) for x in dl):
                                row["cls"] = "a"
                            else:
                                row["cls"] = "ab"
                    else:
                        row["twin"] = a
                        if isinstance(b, Exception):
                            row["entry"] = "raise: " + str(b)[:120]
                            row["cls"] = "REVERSE"
                        else:
                            row["entry"] = b
                            same = all(_same(a[k], b[k], ulps) for k in a)
                            row["cls"] = "=" if same else "DIFF"
                    # generic guards on the input row
                    isinf = v is not None and isinstance(v, float) and math.isinf(v)
                    isnan = v is None or (isinstance(v, float) and math.isnan(v))
                    g1 = isinf
                    g2 = isinf or (isnan and not allow_nan)
                    g3 = (not no_val) and g2
                    row["g"] = [g1, g2, g3]
                    row["dep"] = sorted(dep)
                    prow and None
                    rec["rows"].append(row)
            out.append(rec)
            print(label, f"{time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    with open(os.path.join(HERE, "probe.json"), "w") as fh:
        json.dump(out, fh, default=repr, indent=0)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
