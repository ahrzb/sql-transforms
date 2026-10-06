import sys; sys.path.insert(0, sys.argv[1]); import shim  # noqa
import numpy as np, pyarrow as pa, sklearn
from sklearn.preprocessing import OneHotEncoder
from sql_transform import SQLProjection
rng = np.random.default_rng(1)
n = 300_000
T = pa.table({"b": rng.choice(["p", "q"], n).tolist()})
sql = "SELECT t(struct_pack(b := b)).b_q AS o FROM __THIS__"
with sklearn.config_context(sparse_interface="sparray"):
    p = SQLProjection(sql, transformers={"t": OneHotEncoder(drop="if_binary")}).fit(T.slice(0, 1000))
    try:
        o = p.transform(T)
        ref = np.array([1.0 if v == "q" else 0.0 for v in T["b"].to_pylist()])
        print("sparray serve, 300k rows:", np.array_equal(np.array(o["o"].to_pylist()), ref))
    except Exception as e:
        print("sparray serve raises:", str(e)[:200].replace("\n", " "))
try:
    o = p.transform(T.slice(0, 10))
    print("serve outside context ok")
except Exception as e:
    print("same fitted model served outside sparray context raises:", str(e).replace("\n", " ")[60:200])
