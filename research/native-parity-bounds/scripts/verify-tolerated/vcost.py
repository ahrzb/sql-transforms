"""Independent timing of a lane-0 guard, with spellings of my own, and of
the twin at the same width and batch size.

Spellings (lane 0 wrapped in CASE WHEN <g> THEN error() ELSE lane0):
  none      no guard
  inf_or    OR over abs(x_j) = inf                      (rejects +-inf only)
  fin_or    OR over (abs(x_j) = inf OR x_j = NaN)        (rejects +-inf, NaN)
  fin_mul0  (x_0*0.0 + ... + x_k*0.0) <> 0.0             (rejects +-inf, NaN; one compare)
  inf_2lvl  CASE WHEN fin_mul0 THEN (inf_or) ELSE false  (rejects +-inf only; detail only on suspect rows)
Interleaved rounds, median of per-call time; also the twin (PythonTransform)
on 1,000-row calls at the same width.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/tolerated")
import shim  # noqa: F401
import math, statistics, time, warnings
import numpy as np
import pyarrow as pa
from confit import DuckDBInferFn
from confit import sql as S
from confit.oracle import Oracle
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, MinMaxScaler
from threadpoolctl import threadpool_limits
from sql_transform._udf import PythonTransform
from sql_transform.native import _registry, to_native
from sql_transform.native._registry import query

warnings.simplefilter("ignore")
INF = S.lit(math.inf); NAN = S.lit(math.nan); Z = S.lit(0.0)


def g_inf(x):
    g = None
    for xi in x:
        p = S.fn("abs", xi) == INF
        g = p if g is None else g | p
    return g


def g_fin(x):
    g = None
    for xi in x:
        p = (S.fn("abs", xi) == INF) | (xi == NAN)
        g = p if g is None else g | p
    return g


def g_mul0(x):
    s = None
    for xi in x:
        t = xi * Z
        s = t if s is None else s + t
    return s != Z


def g_2lvl(x):
    return S.case(g_mul0(x), g_inf(x)).otherwise(S.lit(False))


SPELL = {"none": None, "inf_or": g_inf, "fin_or": g_fin, "fin_mul0": g_mul0, "inf_2lvl": g_2lvl}


def patched(orig, g):
    def tr(est, x, types):
        out = list(orig(est, x, types))
        if g is not None:
            out[0] = S.case(g(x), S.fn("error", S.lit("tf: non-finite input"))).otherwise(out[0])
        return out
    return tr


def main():
    rng = np.random.default_rng(2024)
    k = 32
    X = rng.normal(0, 3, size=(500, k))
    nrow = 10_000
    rows = pa.table({"__iid": pa.array([0] * nrow, pa.int64()),
                     **{f"x{j}": pa.array(rng.normal(0, 3, nrow)) for j in range(k)}})
    for name, est in [("StandardScaler", StandardScaler().fit(X)),
                      ("MinMaxScaler(clip)", MinMaxScaler(clip=True).fit(X)),
                      ("KBins(ordinal,5,quantile)", KBinsDiscretizer(n_bins=5, encode="ordinal").fit(X))]:
        step = PythonTransform("tf", {0: est}, pa.schema([(f"x{j}", pa.float64()) for j in range(k)]),
                               pa.struct([(f"f{j}", pa.float64()) for j in range(k)]))
        cls = type(est); orig = _registry._CATALOG[cls]
        fns = {}
        for v, g in SPELL.items():
            _registry._CATALOG[cls] = _registry.Entry(patched(orig.translate, g), orig.ulps, orig._bound if hasattr(orig, "_bound") else None) if False else _registry.Entry(patched(orig.translate, g), 0)
            try:
                nat = to_native(step, strict=True)
            finally:
                _registry._CATALOG[cls] = orig
            f = DuckDBInferFn(query(step), row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[nat])
            f.infer_arrow(rows)
            fns[v] = (nat, f)
        T = {v: [] for v in SPELL}
        for _ in range(9):
            for v in SPELL:
                for _ in range(3):
                    a = time.perf_counter(); fns[v][1].infer_arrow(rows); T[v].append(time.perf_counter() - a)
        base = statistics.median(T["none"]) / nrow * 1e6
        print(f"{name}, {k} features, {nrow}-row calls (us/row; delta; ns/feature/row)")
        for v in SPELL:
            m = statistics.median(T[v]) / nrow * 1e6
            q1, q3 = np.percentile(np.array(T[v]) / nrow * 1e6, [25, 75])
            print(f"  {v:<9} {m:6.3f} [{q1:.3f},{q3:.3f}]  {m-base:+.3f}  {1e3*(m-base)/k:+6.1f}")
        # trap behaviour, reading ONE other field (f5), confit vs DuckDB
        probes = [math.inf, -math.inf, math.nan, None, 1.7976931348623157e308, -1.7976931348623157e308, 5e-324, -0.0, 1e308]
        for v in ["inf_or", "fin_or", "fin_mul0", "inf_2lvl"]:
            nat, _ = fns[v]
            sql1 = f"SELECT tf(__iid, {', '.join(f'x{j}' for j in range(k))}).f5 AS f5 FROM __THIS__"
            f1 = DuckDBInferFn(sql1, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[nat])
            rc, rd = [], []
            for p in probes:
                t = rows.slice(0, 1).set_column(4, "x3", pa.array([p], pa.float64()))
                try:
                    f1.infer_arrow(t); rc.append("ok")
                except Exception:  # noqa: BLE001
                    rc.append("ERR")
                with Oracle() as o:
                    nat.register(o); o.load("__THIS__", t)
                    d = o.try_answer(sql1)
                rd.append("ok" if isinstance(d, pa.Table) else "ERR")
            print(f"    trap {v:<9} (read f5, bad x3) confit {rc} duckdb-same={rc == rd}")
        # twin at the same width, 1,000-row calls
        r1k = rows.slice(0, 1000)
        ft = DuckDBInferFn(query(step), row_tables={"__THIS__": r1k.schema}, static_tables={}, udfs=[step])
        ft.infer_arrow(r1k)
        tt = []
        for _ in range(3):
            a = time.perf_counter(); ft.infer_arrow(r1k); tt.append(time.perf_counter() - a)
        print(f"  twin @1000-row calls: {statistics.median(tt)/1000*1e6:.1f} us/row")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
