"""Run the twin (repo PythonTransform, row by row, plus batched transform) on
every case in a pickle, under the env this process was started with.
usage: twin_run.py CASES.pkl OUT.npz"""
import importlib.util, pickle, sys, json, os
import numpy as np, pyarrow as pa
spec = importlib.util.spec_from_file_location(
    "_udf", "/home/user/sql-transforms/packages/sql-transform/sql_transform/_udf.py")
u = importlib.util.module_from_spec(spec); sys.modules["_udf"] = u; spec.loader.exec_module(u)
from threadpoolctl import threadpool_info
from numpy._core._multiarray_umath import __cpu_features__ as cf
cases = pickle.load(open(sys.argv[1], "rb"))
out = {}
for key, c in cases.items():
    est, X = c["est"], c["X"]
    k = np.asarray(est.transform(X[:1])).shape[1]
    pt = u.PythonTransform(name="t", instances={0: est},
                           takes=pa.schema([(f"f{i}", pa.float64()) for i in range(X.shape[1])]),
                           returns=pa.list_(pa.float64(), k) if k > 1 else pa.float64())
    rows = np.array([pt(0, *r) for r in X.tolist()]).reshape(len(X), k)
    out[key + "__row"] = rows
    out[key + "__batch"] = np.asarray(est.transform(X), dtype=np.float64)
    if hasattr(est, "components_") and hasattr(est, "mean_"):
        out[key + "__mc"] = (est.mean_.reshape(1, -1) @ est.components_.T)[0]
np.savez(sys.argv[2], **{k: v for k, v in out.items() if v is not None})
print(json.dumps(dict(blas=[(i["architecture"], i["num_threads"]) for i in threadpool_info() if i["user_api"] == "blas"],
                      avx512=bool(cf.get("AVX512_SKX")), env={k: v for k, v in os.environ.items() if k.startswith(("OPENBLAS", "NPY_"))})))
