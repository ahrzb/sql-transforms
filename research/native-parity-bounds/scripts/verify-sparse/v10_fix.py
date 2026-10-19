import sys; sys.path.insert(0, sys.argv[1]); import shim  # noqa
import warnings; warnings.simplefilter("ignore")
import inspect, textwrap
import numpy as np, pyarrow as pa, sklearn
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, SplineTransformer
import sql_transform._projection as P, sql_transform._udf as U
from sql_transform import SQLProjection

def _dense(out):
    m = sys.modules.get("scipy.sparse")
    return out.toarray() if m is not None and m.issparse(out) else out

def patch(cls, meth, mod, old, new):
    src = textwrap.dedent(inspect.getsource(getattr(cls, meth)))
    assert src.count(old) == 1, (meth, old)
    g = dict(mod.__dict__); g["_dense"] = _dense
    exec(compile(src.replace(old, new), mod.__file__, "exec"), g)
    setattr(cls, meth, g[meth])
FIX = "--nofix" not in sys.argv
if FIX:
    patch(P.SQLProjection, "_fit_step", P, "probe = np.asarray(est.transform(feats[idx][:1]))",
          "probe = np.asarray(_dense(est.transform(feats[idx][:1])))")
    patch(U.PythonTransform, "__call__", U, "row = est.transform([vals])[0]", "row = _dense(est.transform([vals]))[0]")

rng = np.random.default_rng(42)
n = 400
T = pa.table({"c": rng.choice(["red", "green", "blue", "teal"], n).tolist(),
              "x": rng.normal(size=n).tolist(), "y": rng.uniform(-1, 2, n).tolist()})
Q = pa.table({"c": ["red", "zzz", None, "teal", "blue"],
              "x": [-0.0, 0.0, -5e-324, 1e300, -1e300],
              "y": [5.0, -5.0, -0.0, 0.5, 1.99]})
def bits_rows(tbl):
    out = []
    for r in tbl.to_pylist():
        v = r["o"]
        out.append(None if v is None else tuple((k, np.float64(x).view(np.uint64) if x is not None else None) for k, x in v.items()))
    return out
cases = [
    ("OHE", "SELECT t(struct_pack(c := c)) AS o FROM __THIS__", lambda s: OneHotEncoder(handle_unknown="ignore", sparse_output=s)),
    ("KBins", "SELECT t(struct_pack(x := x, y := y)) AS o FROM __THIS__", lambda s: KBinsDiscretizer(n_bins=4, encode="onehot" if s else "onehot-dense", quantile_method="averaged_inverted_cdf")),
    ("Spline d3 const", "SELECT t(struct_pack(y := y)) AS o FROM __THIS__", lambda s: SplineTransformer(degree=3, n_knots=4, extrapolation="constant", sparse_output=s)),
    ("Spline d0 const", "SELECT t(struct_pack(y := y)) AS o FROM __THIS__", lambda s: SplineTransformer(degree=0, n_knots=2, extrapolation="constant", sparse_output=s)),
]
for iface in ("spmatrix", "sparray"):
    with sklearn.config_context(sparse_interface=iface):
        for label, sql, mk in cases:
            res = {}
            for s in (True, False):
                try:
                    p = SQLProjection(sql, transformers={"t": mk(s)}).fit(T)
                    decl = list(p.udfs.values())[0].returns
                    tr = bits_rows(p.transform(Q))
                    inf = [None if r["o"] is None else tuple((k, np.float64(x).view(np.uint64) if x is not None else None) for k, x in r["o"].items()) for r in p.infer_batch(Q.to_pylist())]
                    res[s] = (str(decl), tr, inf)
                except Exception as e:
                    res[s] = (type(e).__name__ + ": " + str(e).replace("\n", " ")[:120],)
            a, b = res[True], res[False]
            if len(a) == 1 or len(b) == 1:
                print(f"[{iface}] {label}: sparse={a[0][:150]} | dense={b[0][:150]}")
            else:
                print(f"[{iface}] {label}: decl_eq={a[0]==b[0]} ({a[0][:60]}) transform_eq={a[1]==b[1]} infer_eq={a[2]==b[2]} transform==infer {a[1]==a[2]}")
# field read
with sklearn.config_context(sparse_interface="spmatrix"):
    try:
        p = SQLProjection("SELECT t(struct_pack(c := c)).c_red AS o FROM __THIS__", transformers={"t": OneHotEncoder()}).fit(T)
        print("field read .c_red:", p.transform(Q.slice(0, 1)).to_pylist())
    except Exception as e:
        print("field read raises", str(e)[:150])
