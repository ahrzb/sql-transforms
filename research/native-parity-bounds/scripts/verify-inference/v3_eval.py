"""V3 eval: label / leaf / branch changes against T (default-env row twin), split
into NEW rows and re-scored TRAINING rows, for N and for each T'."""
import json, pickle, sys, warnings
import numpy as np
from pathlib import Path
V = Path(__file__).resolve().parent
sys.path.insert(0, str(V))
from v3_common import native  # noqa
warnings.filterwarnings("ignore")


def outs(ms, F):
    F32 = F.astype(np.float32)
    o = dict(lr=ms["lr"].predict(F), lrp=ms["lr"].predict_proba(F)[:, 1],
             dt=ms["dt"].predict(F), dtl=ms["dt"].apply(F32),
             rf=ms["rf"].predict(F), rfl=ms["rf"].apply(F32), rfp=ms["rf"].predict_proba(F)[:, 1],
             hgb=ms["hgb"].predict(F), hgbr=ms["hgb"]._raw_predict(F).ravel())
    return o


def cmp(oT, oC, sl):
    g = lambda k: (oT[k][sl] != oC[k][sl])  # noqa
    return dict(lr=int(g("lr").sum()), lr_dp=float(np.abs(oT["lrp"][sl] - oC["lrp"][sl]).max()),
                dt_lab=int(g("dt").sum()), dt_leaf=int(g("dtl").sum()),
                rf_lab=int(g("rf").sum()), rf_leafrows=int(g("rfl").any(axis=1).sum()),
                hgb_lab=int(g("hgb").sum()), hgb_rows=int(g("hgbr").sum()))


def main(only=None):
    cases = pickle.load(open(V / "data" / "v3_cases.pkl", "rb"))
    fitted = pickle.load(open(V / "data" / "v3_fitted.pkl", "rb"))
    Z = {e: np.load(V / "data" / f"v3_{e}.npz") for e in ["default", "noavx512", "avx2box", "sandybridge"]}
    res = {}
    for name, c in cases.items():
        if only and name not in only:
            continue
        est, X = c["est"], c["X"]
        f = fitted[name]
        ms, nn, ntr = f["models"], f["n_new"], f["n_tr"]
        T = Z["default"][name + "__row"]
        comps = {"N": native(name, est, X), "batch(same box)": Z["default"][name + "__batch"],
                 "noAVX512-numpy/row": Z["noavx512"][name + "__row"],
                 "avx2box/row": Z["avx2box"][name + "__row"], "sandybridge/row": Z["sandybridge"][name + "__row"]}
        Ffit = T.copy(); Ffit[nn:] = f["Ftr"]
        comps["fit_transform(train)"] = Ffit
        oT = outs(ms, T)
        r = {}
        for cn, C in comps.items():
            oC = outs(ms, C)
            d = dict(entries_diff_new=float(np.mean(C[:nn] != T[:nn])), entries_diff_train=float(np.mean(C[nn:] != T[nn:])),
                     max_abs=float(np.abs(C - T).max()),
                     new=cmp(oT, oC, slice(0, nn)), train=cmp(oT, oC, slice(nn, None)))
            # against what the models were TRAINED on (fit_transform) for the training rows
            oF = outs(ms, Ffit)
            d["train_vs_fit"] = cmp(oF, oC, slice(nn, None))
            r[cn] = d
        res[name] = r
        print(f"== {name}  new={nn} train={ntr}", flush=True)
        for cn, d in r.items():
            print(f"  {cn:22s} diff new {d['entries_diff_new']:.3f} train {d['entries_diff_train']:.3f} max|d| {d['max_abs']:.1e} | "
                  f"NEW {d['new']} | TRAIN {d['train']} | TRAIN-vs-fit {d['train_vs_fit']}", flush=True)
    json.dump(res, open(V / "data" / ("v3_results.json" if not only else "v3_results_part.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
