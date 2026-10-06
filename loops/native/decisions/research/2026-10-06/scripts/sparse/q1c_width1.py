"""Q1c: the width-1 sparse case (OneHotEncoder(drop='if_binary') on one
binary feature) under both sparse interfaces, through SQLProjection."""
import sys
sys.path.insert(0, sys.argv[1])
import _shim  # noqa
import pyarrow as pa
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sql_transform import SQLProjection

T = pa.table({"c": ["a", "b", "a", "b"], "name": ["w", "x", "y", "z"]})
for iface in ("spmatrix", "sparray"):
    with sklearn.config_context(sparse_interface=iface):
        p = SQLProjection("SELECT t(struct_pack(c := c)) AS o, name FROM __THIS__",
                          transformers={"t": OneHotEncoder(drop="if_binary")}).fit(T)
        print(iface, "declared", p.udfs["__cf_tf0"].returns)
        try:
            print("   transform ->", p.transform(T).to_pylist())
        except Exception as e:
            print("   transform raises", type(e).__name__, str(e).splitlines()[0][:160])
