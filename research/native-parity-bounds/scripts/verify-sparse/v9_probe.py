import sys; sys.path.insert(0, sys.argv[1]); import shim  # noqa
import warnings; warnings.simplefilter("ignore")
import inspect, textwrap, dataclasses
import numpy as np, pyarrow as pa, scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
import sql_transform.native as N
from sql_transform.native import encode as E, _registry as R
from sql_transform import PythonTransform
src = inspect.getsource(E._encode).splitlines()
src = [l for l in src if not l.startswith("@translates")]
i = next(k for k, l in enumerate(src) if "sparse_output" in l)
del src[i:i + 2]
ns = dict(E.__dict__); exec("\n".join(src), ns)
unguarded = ns["_encode"]
old = R._CATALOG[OneHotEncoder]
R._CATALOG[OneHotEncoder] = dataclasses.replace(old, translate=unguarded) if dataclasses.is_dataclass(old) else old
if not dataclasses.is_dataclass(old):
    old.translate = unguarded
X = np.array([["r"], ["g"], ["b"], ["r"]], dtype=object)
for sparse in (True, False):
    est = OneHotEncoder(handle_unknown="ignore", sparse_output=sparse).fit(X)
    print("sparse" if sparse else "dense ", "probe:", E._probe(est, ["b"], 0, "g"), E._probe(est, ["b"], 0, "zz"))
    names = list(est.get_feature_names_out(["c"]))
    step = PythonTransform("tf", {0: est}, pa.schema([("c", pa.string())]), pa.struct([(n, pa.float64()) for n in names]))
    try:
        fn = N.to_native(step, strict=True)
        print("   to_native ok:", type(fn).__name__)
    except Exception as e:
        print("   to_native raises", type(e).__name__, str(e)[:200])
