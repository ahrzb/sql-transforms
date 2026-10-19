"""Splits learned inside rounding noise: for each tree split (lane f,
threshold t), the half gap g between t and the nearest training value
(fit_transform output) on its node... approximated over the whole training
lane, against delta_f = the largest |C - T| on lane f over every comparator
(N and all T'). A split with g <= delta_f can route a row differently under
a rounding; g >> delta_f cannot except for rows whose own value sits within
delta of t (the density term)."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

from common import FEATS, PLAN, load_case  # noqa: E402

RES = HERE / "results"


def splits_of(models):
    out = []
    t = models["dtree"].tree_
    for i in range(t.node_count):
        if t.children_left[i] != -1:
            out.append(("dtree", int(t.feature[i]), float(t.threshold[i])))
    for e in models["rf"].estimators_:
        t = e.tree_
        m = t.children_left != -1
        out += [("rf", int(f), float(th)) for f, th in zip(t.feature[m], t.threshold[m])]
    for it in models["hgb"]._predictors:
        for p in it:
            for nd in p.nodes:
                if not nd["is_leaf"]:
                    out.append(("hgb", int(nd["feature_idx"]), float(nd["num_threshold"])))
    return out


def main():
    rows = []
    for fam in PLAN:
        for dn in PLAN[fam][0]:
            c = load_case(fam, dn)
            Ftr = c["Ftr"]
            T = np.load(FEATS / "default" / f"{fam}__{dn}__row.npy")
            res = json.loads((RES / f"{fam}__{dn}.json").read_text())
            delta = np.zeros(T.shape[1])
            for name, cc in res["comps"].items():
                delta = np.maximum(delta, np.array(cc["feat"]["lane_max_abs"]))
            sorted_lanes = [np.unique(Ftr[:, f]) for f in range(Ftr.shape[1])]
            sorted_test = [np.sort(T[:, f]) for f in range(T.shape[1])]
            cnt = {"dtree": [0, 0, 0], "rf": [0, 0, 0], "hgb": [0, 0, 0]}
            for model, f, t in splits_of(c["models"]):
                v = sorted_lanes[f]
                j = np.searchsorted(v, t)
                lo = v[j - 1] if j > 0 else -np.inf
                hi = v[j] if j < len(v) else np.inf
                g = min(t - lo, hi - t)
                cnt[model][0] += 1
                if g <= 10 * delta[f]:
                    cnt[model][1] += 1
                # test rows within delta_f of t
                s = sorted_test[f]
                k = np.searchsorted(s, t + delta[f], "right") - np.searchsorted(s, t - delta[f], "left")
                cnt[model][2] += int(k)
            rows.append((fam, dn, cnt))
            print(fam, dn, {m: dict(splits=v[0], noise_splits=v[1], test_rows_within_delta=v[2])
                            for m, v in cnt.items()}, flush=True)
    (RES / "noise_splits.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
