"""Independent re-probe of the tolerated-difference claims.

Design differs from the report's probe.py:
- fit data: 4 features (1 for 1-D), n=60, a different distribution and seed
  (mode "alt"), or the report's own design (mode "same") to cross-check its
  classification logic;
- base row: a real fitted row (markers replaced), not the median;
- lane dependency is found with the ENTRY over random finite values, and
  cross-checked with the twin;
- all rows served one at a time (no batch fallback), twin and entry.

Writes vprobe_<mode>.json.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/tolerated")
import shim  # noqa: F401  (pydantic / py3.14rc2 kwarg)

import json, math, time, warnings
import numpy as np
import pyarrow as pa
from confit import DuckDBInferFn
from threadpoolctl import threadpool_limits
import sklearn

from sql_transform._udf import PythonTransform
from sql_transform.native import NotNative, bound, to_native
from sql_transform.native._check import _same
from sql_transform.native._registry import query
from sql_transform.native.catalog_test import FIXTURES, _runs

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MODE = sys.argv[1] if len(sys.argv) > 1 else "alt"

PROBES = [("+inf", math.inf), ("-inf", -math.inf), ("NaN", math.nan), ("NULL", None),
          ("1e308", 1e308), ("-1e308", -1e308), ("0", 0.0), ("-1", -1.0),
          ("1e6", 1e6), ("-1e6", -1e6), ("0.5", 0.5)]


def design(factory, mode):
    proto_runs = _runs(factory())
    proto = proto_runs[0]
    tags = proto.__sklearn_tags__()
    one_d = not tags.input_tags.two_d_array
    cat = tags.input_tags.categorical
    reg = proto_runs[-1].__sklearn_tags__().estimator_type == "regressor"
    if mode == "same":
        rng = np.random.default_rng(0)
        d = 1 if one_d else 3
        n = 40
        X = (rng.integers(1, 4, size=(n, d)).astype(float) if cat
             else np.round(rng.uniform(1, 10, size=(n, d)), 2))
        if hasattr(proto, "missing_values"):
            mv = float(proto.missing_values)
            X[rng.random(n) < 0.2, 0] = mv
            X[0, 0] = mv
        y = (rng.random(n) < 0.5).astype(int); y[:4] = (0, 1, 0, 1)
        if reg or one_d:
            y = np.nan_to_num(X[:, 0]) + rng.normal(0, 1, n)
        return X, y, d
    rng = np.random.default_rng(12345)
    d = 1 if one_d else 4
    n = 60
    if cat:
        X = rng.integers(1, 5, size=(n, d)).astype(float)
    else:
        X = np.round(rng.gamma(2.0, 3.0, size=(n, d)) + 0.25, 3)
    if hasattr(proto, "missing_values"):
        mv = float(proto.missing_values)
        X[rng.random(n) < 0.25, 0] = mv
        X[1, 0] = mv
    y = (rng.random(n) < 0.5).astype(int); y[:4] = (1, 0, 1, 0)
    if reg or one_d:
        y = 2.0 * np.nan_to_num(X[:, 0]) + rng.normal(0, 2, n)
    return X, y, d


def fin(v):
    return isinstance(v, float) and math.isfinite(v)


class Srv:
    def __init__(self, step, fn, d):
        sch = pa.schema([pa.field("__iid", pa.int64())] + [pa.field(f"x{j}", pa.float64()) for j in range(d)])
        self.f = DuckDBInferFn(query(step, "__iid"), row_tables={"__THIS__": sch}, static_tables={}, udfs=[fn])
        self.d = d

    def one(self, row):
        t = pa.table({"__iid": pa.array([0], pa.int64()),
                      **{f"x{j}": pa.array([row[j]], pa.float64()) for j in range(self.d)}})
        try:
            return self.f.infer_arrow(t).to_pylist()[0]
        except Exception as e:  # noqa: BLE001
            return e


def main():
    out = []
    t0 = time.time()
    rs = np.random.default_rng(7)
    for cls, facs in FIXTURES.items():
        for j, fac in enumerate(facs):
            label = f"{cls.__name__}[{j}]"
            try:
                X, y, d = design(fac, MODE)
                est = fac().fit(X, y)
                w = np.asarray(est.transform(X[:1])).reshape(1, -1).shape[1]
                ret = pa.float64() if w == 1 else pa.struct([(f"f{k}", pa.float64()) for k in range(w)])
                step = PythonTransform("tf", {0: est}, pa.schema([(f"x{k}", pa.float64()) for k in range(d)]), ret)
                nat = to_native(step, strict=True)
                ul = bound(step)
                tw, nv = Srv(step, step, d), Srv(step, nat, d)
            except Exception as e:  # noqa: BLE001
                out.append({"cfg": label, "error": f"{type(e).__name__}: {e}"[:200]})
                continue
            # base: a real fitted row; replace the marker if any
            base = [float(v) for v in X[min(5, len(X) - 1)]]
            r0 = _runs(est)[0]
            if hasattr(r0, "missing_values"):
                mv = float(r0.missing_values)
                base = [(b + 1.0 if (b == mv or (math.isnan(mv) and math.isnan(b))) else b) for b in base]
                base = [1.0 if math.isnan(b) else b for b in base]
            rec = {"cfg": label, "cls": cls.__name__, "ulps": ul, "rows": []}
            for p in range(d):
                # lane dependency from the entry and the twin, on fitted values
                vals = sorted(set(float(v) for v in X[:, p] if not math.isnan(v)))
                pick = list(rs.choice(vals, size=min(8, len(vals)), replace=False))
                dep_e, dep_t = set(), set()
                seen_e, seen_t = [], []
                for v in pick:
                    r = list(base); r[p] = float(v)
                    a = nv.one(r); b = tw.one(r)
                    if isinstance(a, dict): seen_e.append(a)
                    if isinstance(b, dict): seen_t.append(b)
                for seen, dep in ((seen_e, dep_e), (seen_t, dep_t)):
                    if seen:
                        for k in seen[0]:
                            if len({repr(s[k]) for s in seen}) > 1:
                                dep.add(k)
                for name, v in PROBES:
                    r = list(base); r[p] = v
                    a = tw.one(r); b = nv.one(r)
                    row = {"feat": p, "probe": name, "dep_e": sorted(dep_e), "dep_t": sorted(dep_t)}
                    if isinstance(a, Exception):
                        row["twin"] = "raise: " + str(a).strip().split("\n")[-1][-140:]
                        if isinstance(b, Exception):
                            row["entry"] = "raise"; row["k"] = "c"
                        else:
                            row["entry"] = b
                            lanes = list(b.values())
                            row["all_finite"] = all(fin(x) for x in lanes)
                            dl = [b[k] for k in dep_t]
                            row["dep_finite"] = (all(fin(x) for x in dl) if dl else None)
                            row["dep_any_finite"] = (any(fin(x) for x in dl) if dl else None)
                            vv = [math.nan if q is None else q for q in r]
                            try:
                                with sklearn.config_context(assume_finite=True), np.errstate(all="ignore"):
                                    ref = [float(z) for z in np.asarray(est.transform([vv]), dtype=float).reshape(-1)]
                                row["ref"] = ref
                                row["ref_same"] = all(_same(rv, ev, ul) for rv, ev in zip(ref, lanes))
                            except Exception as e:  # noqa: BLE001
                                row["ref"] = "raise: " + str(e).split("\n")[0][-100:]
                                row["ref_same"] = None
                            row["k"] = "answer"
                    else:
                        row["twin"] = a
                        if isinstance(b, Exception):
                            row["entry"] = "raise: " + str(b)[:100]; row["k"] = "REVERSE"
                        else:
                            row["entry"] = b
                            row["k"] = "=" if all(_same(a[k], b[k], ul) for k in a) else "DIFF"
                    rec["rows"].append(row)
            out.append(rec)
            print(label, f"{time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    json.dump(out, open(os.path.join(HERE, f"vprobe_{MODE}.json"), "w"), default=repr)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
