"""Q1b: the failure through the real fit path (SQLProjection._fit_step's
np.asarray probe) and the serve path, for spmatrix and for sklearn's
sparse_interface='sparray'."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/sparse"); import _shim  # noqa
import traceback

import numpy as np
import pyarrow as pa
import sklearn
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from sql_transform import SQLProjection

TRAIN = pa.table(
    {
        "color": ["red", "blue", "red", "green"],
        "age": [40.0, 30.0, 20.0, 25.0],
        "name": ["x", "y", "z", "w"],
    }
)


def run(label, sql, est, iface="spmatrix"):
    print(f"== {label} [sparse_interface={iface}]")
    with sklearn.config_context(sparse_interface=iface):
        try:
            p = SQLProjection(sql, transformers={"t": est}).fit(TRAIN)
        except Exception as e:
            print("  FIT raises:", type(e).__name__, str(e)[:300])
            return
        (u,) = p.udfs.values()
        print("  fit OK; declared returns:", u.returns)
        try:
            out = p.transform(TRAIN)
            print("  transform (DuckDB batch) ->", out.to_pylist()[:2])
        except Exception as e:
            msg = str(e).replace("\n", " ")
            print("  transform raises:", type(e).__name__, msg[:300])
        try:
            r = p.infer(TRAIN.slice(0, 1).to_pylist()[0])
            print("  infer (confit row) ->", r)
        except Exception as e:
            msg = str(e).replace("\n", " ")
            print("  infer raises:", type(e).__name__, msg[:300])


for iface in ("spmatrix", "sparray"):
    run("OHE() whole struct", "SELECT t(struct_pack(c := color)) AS o, name FROM __THIS__",
        OneHotEncoder(handle_unknown="ignore"), iface)
    run("OHE() field read .c_red", "SELECT t(struct_pack(c := color)).c_red AS o, name FROM __THIS__",
        OneHotEncoder(handle_unknown="ignore"), iface)
    run("KBins(onehot) whole struct", "SELECT t(struct_pack(a := age)) AS o, name FROM __THIS__",
        KBinsDiscretizer(n_bins=2, encode="onehot", quantile_method="averaged_inverted_cdf"), iface)
run("OHE(sparse_output=False) control", "SELECT t(struct_pack(c := color)) AS o, name FROM __THIS__",
    OneHotEncoder(handle_unknown="ignore", sparse_output=False))
