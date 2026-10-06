"""Twin features under the CURRENT process env (set before import by the
caller): row by row through the repo's PythonTransform (est.transform([vals])
per row, as served), and one batched est.transform(X_test).

usage: feats.py ENVNAME [family ...]"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402

from common import FEATS, PLAN, load_case  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    env = sys.argv[1]
    fams = sys.argv[2:] or list(PLAN)
    from numpy._core._multiarray_umath import __cpu_features__ as cf
    from threadpoolctl import threadpool_info

    # the repo's own PythonTransform (_udf.py has no package imports; the
    # package __init__ does not import on this Python 3.14rc2 + pydantic)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_udf", "/home/user/sql-transforms/packages/sql-transform/sql_transform/_udf.py")
    _udf = importlib.util.module_from_spec(spec)
    sys.modules["_udf"] = _udf
    spec.loader.exec_module(_udf)
    PythonTransform = _udf.PythonTransform

    out = FEATS / env
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(
        env={k: v for k, v in os.environ.items() if k.startswith(("OPENBLAS", "NPY_", "OMP"))},
        blas=[(i["prefix"], i["architecture"], i["num_threads"]) for i in threadpool_info()
              if i["user_api"] == "blas"],
        npy_avx512_skx=bool(cf.get("AVX512_SKX")),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(env, meta, flush=True)
    for fam in fams:
        for dn in PLAN[fam][0]:
            t0 = time.time()
            c = load_case(fam, dn)
            est, X = c["est"], c["Xte"]
            k = c["Ftr"].shape[1]
            takes = pa.schema([(f"f{i}", pa.float64()) for i in range(X.shape[1])])
            ret = pa.float64() if k == 1 else pa.list_(pa.float64(), k)
            pt = PythonTransform(name="t", instances={0: est}, takes=takes, returns=ret)
            rows = np.empty((len(X), k))
            for r, xr in enumerate(X.tolist()):
                rows[r] = pt(0, *xr)
            batch = np.asarray(est.transform(X), dtype=np.float64)
            np.save(out / f"{fam}__{dn}__row.npy", rows)
            np.save(out / f"{fam}__{dn}__batch.npy", batch)
            print(f"{env} {fam} {dn} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
