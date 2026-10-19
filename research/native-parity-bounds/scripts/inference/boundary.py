"""Stress test of the flip model: put rows ON a decision boundary (point mass
at the threshold), where P(flip) = rho*E|delta| is O(1), and compare how often
N and T' disagree with T there.

Boundary rows: for pairs (x_a, x_b) of test rows on opposite sides of a
predicate, bisect alpha on x(alpha) = (1-alpha) x_a + alpha x_b with the
twin's batched transform (default env) until alpha_lo, alpha_hi are adjacent
doubles; keep x(alpha_lo) and x(alpha_hi).

usage: boundary.py make | boundary.py feats ENV | boundary.py eval"""

import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

from common import (  # noqa: E402
    DATA, FEATS, load_case, native_chi2, native_kmeans, native_pca, native_yj, native_kmeans_direct,
)

warnings.filterwarnings("ignore")
CASES = [("pca_95", "breast"), ("pca_all", "synth_redund"), ("kmeans8", "breast"),
         ("yj_std", "breast"), ("chi2", "breast")]
NPAIRS = 1000


def predicates(c):
    m = c["models"]
    preds = {}
    preds["logreg"] = lambda F: m["logreg"].predict(F) == m["logreg"].classes_[0]
    t = m["dtree"].tree_
    f0, t0 = int(t.feature[0]), float(t.threshold[0])
    preds["dtree_root"] = lambda F: F[:, f0].astype(np.float32) <= t0
    nd = m["hgb"]._predictors[0][0].nodes[0]
    fh, th = int(nd["feature_idx"]), float(nd["num_threshold"])
    preds["hgb_root"] = lambda F: F[:, fh] <= th
    if c["family"].startswith("kmeans"):
        preds["argmin"] = lambda F: np.argmin(F, axis=1) == 0
    return preds


def make():
    rng = np.random.default_rng(1)
    out = {}
    for fam, dn in CASES:
        c = load_case(fam, dn)
        est, X = c["est"], c["Xte"][:50000]
        F = est.transform(X)
        for pname, P in predicates(c).items():
            p = P(F)
            A, B = np.nonzero(p)[0], np.nonzero(~p)[0]
            if len(A) == 0 or len(B) == 0:
                continue
            xa = X[rng.choice(A, NPAIRS)]
            xb = X[rng.choice(B, NPAIRS)]
            lo = np.zeros(NPAIRS)
            hi = np.ones(NPAIRS)
            for _ in range(80):
                mid = lo + (hi - lo) / 2
                xm = (1 - mid)[:, None] * xa + mid[:, None] * xb
                pm = P(est.transform(xm))
                lo = np.where(pm, mid, lo)
                hi = np.where(pm, hi, mid)
            x_lo = (1 - lo)[:, None] * xa + lo[:, None] * xb
            x_hi = (1 - hi)[:, None] * xa + hi[:, None] * xb
            Xb = np.vstack([x_lo, x_hi])
            if fam == "chi2":
                Xb = np.maximum(Xb, 0.0)
            out[f"{fam}__{dn}__{pname}"] = Xb
            print(fam, dn, pname, Xb.shape, flush=True)
    np.savez(DATA / "boundary.npz", **out)


def feats(env):
    import importlib.util

    import pyarrow as pa

    spec = importlib.util.spec_from_file_location(
        "_udf", "/home/user/sql-transforms/packages/sql-transform/sql_transform/_udf.py")
    u = importlib.util.module_from_spec(spec)
    sys.modules["_udf"] = u
    spec.loader.exec_module(u)
    Z = np.load(DATA / "boundary.npz")
    res = {}
    for key in Z.files:
        fam, dn, pname = key.split("__")
        c = load_case(fam, dn)
        est, X = c["est"], Z[key]
        k = c["Ftr"].shape[1]
        pt = u.PythonTransform(name="t", instances={0: est},
                               takes=pa.schema([(f"f{i}", pa.float64()) for i in range(X.shape[1])]),
                               returns=pa.list_(pa.float64(), k))
        res[key + "__row"] = np.array([pt(0, *r) for r in X.tolist()])
        res[key + "__batch"] = est.transform(X)
    (FEATS / env).mkdir(parents=True, exist_ok=True)
    np.savez(FEATS / env / "boundary.npz", **res)
    print(env, "done")


def native(fam, est, X):
    if fam.startswith("pca"):
        return native_pca(est, X)[0]
    if fam.startswith("kmeans"):
        return native_kmeans(est, X)[0]
    if fam.startswith("yj"):
        return native_yj(est, X)[0]
    return native_chi2(est, X)[0]


def evaluate():
    Z = np.load(DATA / "boundary.npz")
    envs = ["default", "haswell", "sandybridge", "npy_noavx512", "avx2_box"]
    F = {e: np.load(FEATS / e / "boundary.npz") for e in envs if (FEATS / e / "boundary.npz").exists()}
    rows = []
    for key in Z.files:
        fam, dn, pname = key.split("__")
        c = load_case(fam, dn)
        P = predicates(c)[pname]
        T = F["default"][key + "__row"]
        pT = P(T)
        r = dict(case=f"{fam}/{dn}", target=pname, n=len(T))
        r["N"] = int((P(native(fam, c["est"], Z[key])) != pT).sum())
        if fam.startswith("kmeans"):
            r["N_direct"] = int((P(native_kmeans_direct(c["est"], Z[key])) != pT).sum())
        r["default/batch"] = int((P(F["default"][key + "__batch"]) != pT).sum())
        for e in envs[1:]:
            if e in F:
                r[f"{e}/row"] = int((P(F[e][key + "__row"]) != pT).sum())
        rows.append(r)
        print(json.dumps(r), flush=True)
    (HERE / "results" / "boundary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make()
    elif sys.argv[1] == "feats":
        feats(sys.argv[2])
    else:
        evaluate()
