import sys; sys.path.insert(0, sys.argv[1]); import shim  # noqa
import numpy as np, pyarrow as pa, sklearn, scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sql_transform import SQLProjection, PythonTransform

rng = np.random.default_rng(7)
n = 5000
T = pa.table({"k": rng.choice(["a", "b", "c", "d"], n).tolist(),
              "x": rng.normal(size=n).tolist(),
              "bin": rng.choice(["p", "q"], n).tolist()})

print("np.asarray(csr):", np.asarray(sp.csr_matrix(np.eye(3)[:1])).shape, np.asarray(sp.csr_matrix(np.eye(3)[:1])).dtype)
print("np.asarray(csr_array):", np.asarray(sp.csr_array(np.eye(3)[:1])).shape)
for exc_try in (sp.csr_matrix(np.eye(3)[:1]), sp.csr_array(np.eye(3)[:1])):
    try:
        np.asarray(exc_try, dtype=np.float64)
        print("asarray dtype ok")
    except Exception as e:
        print("asarray(dtype=float64) on", type(exc_try).__name__, "->", type(e).__name__, str(e)[:100])

def go(label, sql, est, iface):
    with sklearn.config_context(sparse_interface=iface):
        try:
            p = SQLProjection(sql, transformers={"t": est}).fit(T)
        except Exception as e:
            print(f"[{iface}] {label}: FIT {type(e).__name__}: {str(e)[:160]}"); return
        u = list(p.udfs.values())[0]
        res = []
        try:
            o = p.transform(T); res.append(f"transform ok {o.to_pylist()[0]}")
        except Exception as e:
            m = str(e).replace("\n", " "); i = m.find("Error:", 30)
            res.append("transform " + type(e).__name__ + ": " + m[m.find("exception occurred"):][:150])
        try:
            r = p.infer(T.slice(0, 1).to_pylist()[0]); res.append(f"infer ok {r}")
        except Exception as e:
            res.append("infer " + type(e).__name__ + ": " + str(e)[:140])
        print(f"[{iface}] {label}: declared {u.returns}\n    " + "\n    ".join(res))

for iface in ("spmatrix", "sparray"):
    go("OHE whole", "SELECT t(struct_pack(k := k)) AS o FROM __THIS__", OneHotEncoder(), iface)
    go("OHE field", "SELECT t(struct_pack(k := k)).k_b AS o FROM __THIS__", OneHotEncoder(), iface)
    go("KBins uniform onehot", "SELECT t(struct_pack(x := x)) AS o FROM __THIS__", KBinsDiscretizer(n_bins=3, strategy="uniform"), iface)
    go("OHE if_binary k=1", "SELECT t(struct_pack(b := bin)) AS o FROM __THIS__", OneHotEncoder(drop="if_binary"), iface)
    go("OHE if_binary k=1 dense", "SELECT t(struct_pack(b := bin)) AS o FROM __THIS__", OneHotEncoder(drop="if_binary", sparse_output=False), iface)
