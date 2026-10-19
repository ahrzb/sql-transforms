"""Downstream effect of each feature version against T (the twin, row by row,
default kernel). All downstream models run here, in ONE process (default
env), on full same-shape arrays, so any output difference comes from the
features alone.

usage: analyze.py family [dataset ...]   -> results/<family>__<ds>.json"""

import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common import EPS, FEATS, PLAN, load_case, ulps  # noqa: E402

warnings.filterwarnings("ignore")
RES = HERE / "results"
ENVS = ["default", "default_rerun", "threads1", "haswell", "sandybridge",
        "npy_noavx512", "avx2_box", "sse_box"]
K_NEAR = 16


def comparators(fam, dn, n_tr):
    out = {}
    nat = FEATS / "native"
    out["N"] = (np.load(nat / f"{fam}__{dn}__N.npy"), None)
    if fam.startswith("kmeans"):
        out["N_direct"] = (np.load(nat / f"{fam}__{dn}__direct.npy"), None)
    for env in ENVS:
        for mode in ("row", "batch"):
            if env == "default" and mode == "row":
                continue
            p = FEATS / env / f"{fam}__{dn}__{mode}.npy"
            if p.exists():
                out[f"{env}/{mode}"] = (np.load(p), None)
    return out


# --------------------------------------------------------------- trees
def sk_tree_nodes(tree, X32):
    """(feature, threshold, idx_near[k], h) per internal node of a fitted
    sklearn tree_, rows routed as sklearn does: float32(x) <= threshold."""
    t = tree
    res = []
    stack = [(0, np.arange(len(X32)))]
    while stack:
        node, idx = stack.pop()
        f = t.feature[node]
        if t.children_left[node] == -1 or len(idx) == 0:
            continue
        thr = t.threshold[node]
        x = X32[idx, f]
        go_left = x <= thr
        dist = np.abs(x.astype(np.float64) - thr)
        k = min(K_NEAR, len(idx))
        near = np.argpartition(dist, k - 1)[:k]
        res.append((f, thr, idx[near], dist[near].max()))
        stack.append((t.children_left[node], idx[go_left]))
        stack.append((t.children_right[node], idx[~go_left]))
    return res


def hgb_nodes(pred_nodes, X):
    res = []
    stack = [(0, np.arange(len(X)))]
    while stack:
        node, idx = stack.pop()
        nd = pred_nodes[node]
        if nd["is_leaf"] or len(idx) == 0:
            continue
        f = int(nd["feature_idx"])
        thr = float(nd["num_threshold"])
        x = X[idx, f]
        go_left = x <= thr
        dist = np.abs(x - thr)
        k = min(K_NEAR, len(idx))
        near = np.argpartition(dist, k - 1)[:k]
        res.append((f, thr, idx[near], dist[near].max()))
        stack.append((int(nd["left"]), idx[go_left]))
        stack.append((int(nd["right"]), idx[~go_left]))
    return res


def expected_straddles(nodes, absdelta):
    """sum over splits of (rows per unit x at the threshold) * E|delta| near it:
    k/(2h) * mean |delta| of the k rows nearest the threshold."""
    tot = 0.0
    for f, thr, near, h in nodes:
        md = absdelta[near, f].mean()
        if md == 0:
            continue
        if h == 0:
            tot += len(near)  # rows sitting exactly on the threshold
        else:
            tot += len(near) / (2 * h) * md
    return tot


def margin_expected(margin, dmargin):
    """one-sided margin >= 0 (or |signed|): k/(2h) * E|dmargin| near 0."""
    m = np.abs(margin)
    k = K_NEAR
    near = np.argpartition(m, k - 1)[:k]
    h = m[near].max()
    md = np.abs(dmargin[near]).mean()
    return (len(near) / (2 * h) * md) if h > 0 else float("inf"), float(h)


# --------------------------------------------------------------- outputs
def outputs(models, F, kmeans_like):
    o = {}
    F32 = F.astype(np.float32)
    m = models
    o["logreg_pred"] = m["logreg"].predict(F)
    o["logreg_proba"] = m["logreg"].predict_proba(F)
    o["logreg_dec"] = m["logreg"].decision_function(F)
    o["linreg"] = m["linreg"].predict(F)
    o["dtree_pred"] = m["dtree"].predict(F)
    o["dtree_leaf"] = m["dtree"].apply(F32)
    o["dtreg_leaf"] = m["dtree_reg"].apply(F32)
    o["dtreg"] = m["dtree_reg"].predict(F)
    o["rf_pred"] = m["rf"].predict(F)
    o["rf_proba"] = m["rf"].predict_proba(F)
    o["rf_leaf"] = m["rf"].apply(F32)
    o["hgb_pred"] = m["hgb"].predict(F)
    o["hgb_proba"] = m["hgb"].predict_proba(F)
    o["hgb_raw"] = m["hgb"]._raw_predict(F)
    if kmeans_like:
        o["argmin"] = np.argmin(F, axis=1)
        s = np.sort(F, axis=1)
        o["kmargin"] = s[:, 1] - s[:, 0]
    return o


def maxabs_ulps(a, b):
    d = np.abs(a - b)
    u = ulps(a, b)
    return float(d.max()), float(u.max())


def compare(oT, oC, n):
    r = {}
    r["logreg_flips"] = int((oT["logreg_pred"] != oC["logreg_pred"]).sum())
    r["logreg_dproba_abs"], r["logreg_dproba_ulps"] = maxabs_ulps(oT["logreg_proba"], oC["logreg_proba"])
    r["linreg_dabs"], r["linreg_dulps"] = maxabs_ulps(oT["linreg"], oC["linreg"])
    r["dtree_flips"] = int((oT["dtree_pred"] != oC["dtree_pred"]).sum())
    r["dtree_leafflips"] = int((oT["dtree_leaf"] != oC["dtree_leaf"]).sum())
    r["dtreg_leafflips"] = int((oT["dtreg_leaf"] != oC["dtreg_leaf"]).sum())
    r["dtreg_dabs"] = float(np.abs(oT["dtreg"] - oC["dtreg"]).max())
    r["rf_flips"] = int((oT["rf_pred"] != oC["rf_pred"]).sum())
    r["rf_rows_leafflip"] = int((oT["rf_leaf"] != oC["rf_leaf"]).any(axis=1).sum())
    r["rf_dproba_abs"] = float(np.abs(oT["rf_proba"] - oC["rf_proba"]).max())
    r["hgb_flips"] = int((oT["hgb_pred"] != oC["hgb_pred"]).sum())
    r["hgb_rows_rawdiff"] = int((oT["hgb_raw"].reshape(n, -1) != oC["hgb_raw"].reshape(n, -1)).any(axis=1).sum())
    r["hgb_dproba_abs"], r["hgb_dproba_ulps"] = maxabs_ulps(oT["hgb_proba"], oC["hgb_proba"])
    if "argmin" in oT:
        r["argmin_flips"] = int((oT["argmin"] != oC["argmin"]).sum())
    return r


def run_case(fam, dn):
    t0 = time.time()
    c = load_case(fam, dn)
    models = c["models"]
    models["rf"].n_jobs = 4
    T = np.load(FEATS / "default" / f"{fam}__{dn}__row.npy")
    S = np.load(FEATS / "native" / f"{fam}__{dn}__S.npy")
    n_tr = len(c["Xtr"])
    km = fam.startswith("kmeans")
    oT = outputs(models, T, km)
    # tree split structures for the analytic estimate (routing by T)
    T32 = T.astype(np.float32)
    dt_nodes = sk_tree_nodes(models["dtree"].tree_, T32)
    rf_nodes = [sk_tree_nodes(e.tree_, T32) for e in models["rf"].estimators_[:10]]
    hgb_nodes_ = [hgb_nodes(p.nodes, T) for it in models["hgb"]._predictors for p in it]
    # margins
    dec = oT["logreg_dec"]
    if dec.ndim == 1:
        lr_margin = dec
    else:
        s = np.sort(dec, axis=1)
        lr_margin = s[:, -1] - s[:, -2]
    out = dict(family=fam, dataset=dn, n_rows=len(T), lanes=T.shape[1], n_train=n_tr,
               n_dt_splits=len(dt_nodes), comps={})
    comps = comparators(fam, dn, n_tr)
    # the twin's own training/serving skew: fit_transform (what the models
    # were trained on) vs row-by-row transform of the same training rows
    comps["fit_transform(train rows)"] = (None, "ft")
    for name, (C, kind) in comps.items():
        if kind == "ft":
            Tsub = T[-n_tr:]
            C = c["Ftr"]
            Ssub = S[-n_tr:]
            oTs = outputs(models, Tsub, km)
            oC = outputs(models, C, km)
            Tref, Sref, oref = Tsub, Ssub, oTs
        else:
            oC = outputs(models, C, km)
            Tref, Sref, oref = T, S, oT
        d = np.abs(C - Tref)
        u = ulps(C, Tref)
        if km:
            K = np.abs(C.astype(np.float64) ** 2 - Tref ** 2) / (EPS * Sref)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                K = d / (EPS * Sref)
            K = np.where(d == 0, 0.0, K)
        Kf = K[np.isfinite(K)]
        feat = dict(
            entries_diff=float(np.mean(C != Tref)),
            rows_any_diff=int((C != Tref).any(axis=1).sum()),
            max_ulps=float(u.max()),
            p99_ulps=float(np.percentile(u, 99)),
            max_abs=float(d.max()),
            K_max=float(Kf.max()) if Kf.size else 0.0,
            K_p99=float(np.percentile(Kf, 99)) if Kf.size else 0.0,
            f32_diff=float(np.mean(C.astype(np.float32) != Tref.astype(np.float32))),
            lane_max_abs=[float(v) for v in d.max(axis=0)],
        )
        down = compare(oref, oC, len(C))
        # analytic estimates (only meaningful against the full test set)
        if kind != "ft":
            ad = d
            est = dict(
                dtree=expected_straddles(dt_nodes, ad),
                rf=10.0 * sum(expected_straddles(nn, ad) for nn in rf_nodes),
                hgb=sum(expected_straddles(nn, ad) for nn in hgb_nodes_),
            )
            dd = (oC["logreg_dec"] - oT["logreg_dec"])
            if dd.ndim > 1:
                dd = np.abs(dd).max(axis=1) * 2
            est["logreg"], lr_h = margin_expected(lr_margin, dd)
            if km:
                dm = np.abs(oC["kmargin"] - oT["kmargin"])
                est["argmin"], km_h = margin_expected(oT["kmargin"], dm)
                est["argmin_h"] = km_h
            est["logreg_h"] = lr_h
        else:
            est = {}
        out["comps"][name] = dict(feat=feat, down=down, expected=est)
    with open(RES / f"{fam}__{dn}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"{fam} {dn} done {time.time() - t0:.1f}s", flush=True)


def main():
    RES.mkdir(exist_ok=True)
    fam = sys.argv[1]
    dss = sys.argv[2:] or PLAN[fam][0]
    for dn in dss:
        run_case(fam, dn)


if __name__ == "__main__":
    main()
