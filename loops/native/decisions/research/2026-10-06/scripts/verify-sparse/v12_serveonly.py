import sys; sys.path.insert(0, sys.argv[1]); import shim  # noqa
import warnings; warnings.simplefilter("ignore")
import inspect, textwrap, numpy as np, pyarrow as pa
from sklearn.preprocessing import OneHotEncoder
import sql_transform._udf as U
from sql_transform import SQLProjection
def _dense(out):
    m = sys.modules.get("scipy.sparse")
    return out.toarray() if m is not None and m.issparse(out) else out
src = textwrap.dedent(inspect.getsource(U.PythonTransform.__call__)).replace("row = est.transform([vals])[0]", "row = _dense(est.transform([vals]))[0]")
g = dict(U.__dict__); g["_dense"] = _dense; exec(src, g); U.PythonTransform.__call__ = g["__call__"]
T = pa.table({"c": ["a", "b", "c", "a"]})
p = SQLProjection("SELECT t(struct_pack(c := c)) AS o FROM __THIS__", transformers={"t": OneHotEncoder()}).fit(T)
print(list(p.udfs.values())[0].returns)
try: p.transform(T)
except Exception as e: print(str(e).replace("\n", " ")[60:170])
