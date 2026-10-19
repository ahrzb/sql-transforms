"""Q6: the proposed two-site densify, applied by monkeypatch OUTSIDE the repo:
(1) the fit probe (_projection.py:352 `np.asarray(est.transform(...))`),
(2) the serve call (_udf.py:314 `est.transform([vals])[0]`).
Then: sparse-default fits declare the dense config's struct, and serve the
dense config's bits on both bindings (DuckDB batch, confit row)."""
import struct
import sys

sys.path.insert(0, sys.argv[1])
import _shim  # noqa: F401,E402

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import sklearn  # noqa: E402
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, SplineTransformer  # noqa: E402

import sql_transform._udf as U  # noqa: E402
from sql_transform import SQLProjection  # noqa: E402


def _dense(out):
    """The helper the change would add: scipy sparse (either interface, any
    format) -> ndarray via .toarray(), as sklearn's own dense configs do."""
    m = sys.modules.get("scipy.sparse")
    return out.toarray() if m is not None and m.issparse(out) else out


_orig_asarray = np.asarray


def _asarray(a, *args, **kw):  # stands in for site (1)
    return _orig_asarray(_dense(a), *args, **kw)


def _call(self, iid, *feats):  # site (2): PythonTransform.__call__ with _dense
    if iid is None:
        return None
    est = self.instances[iid]
    vals = [U._as_feature(f, t) for f, t in zip(feats, self.take_types, strict=True)]
    row = _dense(est.transform([vals]))[0]
    out = tuple(float(v) for v in U._flatten_row(row))
    if len(out) != len(self.return_types):
        raise U.UDFError("width")
    return out


rng = np.random.default_rng(3)
n = 300
T = pa.table({
    "c": rng.choice(["red", "blue", "green", "teal"], size=n).tolist(),
    "a": rng.normal(size=n).tolist(),
    "b": rng.uniform(-1, 2, size=n).tolist(),
})
P = pa.table({  # probe rows: unknown category, NULL, -0.0, out of range
    "c": ["red", "zzz", None, "teal"],
    "a": [-0.0, 0.0, None, 1e9],
    "b": [-5.0, 0.5, 2.0, -0.0],
})


def bits(rows):
    out = []
    for r in rows:
        for v in r["o"].values():
            out.append(struct.pack("<d", v) if isinstance(v, float) else v)
    return out


np.asarray = _asarray
U.PythonTransform.__call__ = _call
cases = [
    ("OneHotEncoder", "SELECT t(struct_pack(c := c)) AS o FROM __THIS__",
     lambda: OneHotEncoder(handle_unknown="ignore"), lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ("KBinsDiscretizer", "SELECT t(struct_pack(a := a, b := b)) AS o FROM __THIS__",
     lambda: KBinsDiscretizer(encode="onehot", quantile_method="averaged_inverted_cdf"),
     lambda: KBinsDiscretizer(encode="onehot-dense", quantile_method="averaged_inverted_cdf")),
    ("SplineTransformer", "SELECT t(struct_pack(a := a, b := b)) AS o FROM __THIS__",
     lambda: SplineTransformer(sparse_output=True), lambda: SplineTransformer()),
]
try:
    for iface in ("spmatrix", "sparray"):
        with sklearn.config_context(sparse_interface=iface):
            for name, sql, mk_s, mk_d in cases:
                ps = SQLProjection(sql, transformers={"t": mk_s()}).fit(T)
                pd_ = SQLProjection(sql, transformers={"t": mk_d()}).fit(T)
                rs, rd = ps.udfs["__cf_tf0"].returns, pd_.udfs["__cf_tf0"].returns
                probe = P if name == "OneHotEncoder" else (P.slice(0, 2) if name == "KBinsDiscretizer" else pa.concat_tables([P.slice(0, 2), P.slice(3, 1)]))  # NaN rejected by KBins, Spline
                tr = bits(ps.transform(T).to_pylist() + ps.transform(probe).to_pylist()) == \
                    bits(pd_.transform(T).to_pylist() + pd_.transform(probe).to_pylist())
                rows = T.slice(0, 50).to_pylist() + probe.to_pylist()
                inf = bits([ps.infer(r) for r in rows]) == bits([pd_.infer(r) for r in rows])
                print(f"[{iface}] {name}: declared equal={rs == rd} (width {rs.num_fields});"
                      f" DuckDB transform bit-equal={tr}; confit infer bit-equal={inf}")
finally:
    np.asarray = _orig_asarray
