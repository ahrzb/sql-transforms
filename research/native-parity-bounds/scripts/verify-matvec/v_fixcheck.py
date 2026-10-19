"""Regenerate catalog PCA(whiten=True) fixtures for some seeds from the repo
source (own extraction) and compare with the research pickle."""
import ast, importlib.util, sys, pickle, warnings, random, math, os, functools
from typing import Any
import numpy as np, pyarrow as pa
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
REPO = "/home/user/sql-transforms/packages/sql-transform/sql_transform"
spec = importlib.util.spec_from_file_location("u", f"{REPO}/_udf.py"); u = importlib.util.module_from_spec(spec); sys.modules["u"] = u; spec.loader.exec_module(u)
tree = ast.parse(open(f"{REPO}/native/catalog_test.py").read())
keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in {"_fit_matrix","_step","_runs","_draw","_value","_rows"}) or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {"VOCAB","UNSEEN","EDGES","SMALL","MAX_LANES"})]
ns = dict(np=np, pa=pa, random=random, warnings=warnings, math=math, os=os, Any=Any, Pipeline=Pipeline, ColumnTransformer=ColumnTransformer, FeatureUnion=FeatureUnion, PythonTransform=u.PythonTransform)
exec(compile(ast.Module(body=keep, type_ignores=[]), "x", "exec"), ns)
fx = pickle.load(open(sys.argv[1], "rb"))
warnings.simplefilter("ignore")
bad = 0
for seed in range(0, 1000, 23):
    st = ns["_step"](lambda: PCA(whiten=True), seed)
    rows = ns["_rows"](st, seed).to_pylist()
    f = fx[seed]
    assert f["seed"] == seed
    same = rows == f["rows"] and list(st.instances) == list(f["instances"]) and all(
        np.array_equal(st.instances[i].components_, f["instances"][i].components_) and np.array_equal(st.instances[i].mean_, f["instances"][i].mean_)
        and np.array_equal(st.instances[i].explained_variance_, f["instances"][i].explained_variance_) for i in st.instances)
    bad += not same
print("seeds checked", len(range(0,1000,23)), "mismatches", bad)
