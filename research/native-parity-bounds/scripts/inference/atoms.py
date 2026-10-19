"""HGB thresholds are training values (np.percentile 'averaged_inverted_cdf'
returns an order statistic). A row that reproduces the training row whose
value IS the threshold sits a rounding away from it; it changes branch iff
its two computations land on opposite sides of t.

Atom model: for every HGB split (f, t), the training rows i with
Ftr[i, f] == t; every test row that is a copy of such a row and reaches the
node (routed by T) is an atom pair. Predicted branch changes
= sum over atom pairs of P(straddle) ~ 0.5 * P(C != T on lane f) (each side
of t equally likely once the two values differ). Compared with the observed
number of straddles at those pairs and with rows whose HGB raw score changed."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

from common import FEATS, PLAN, load_case  # noqa: E402

CMPS = ["N", "default/batch", "haswell/row", "sandybridge/row", "avx2_box/row"]


def load_cmp(fam, dn, name):
    if name == "N":
        return np.load(FEATS / "native" / f"{fam}__{dn}__N.npy")
    env, mode = name.split("/")
    return np.load(FEATS / env / f"{fam}__{dn}__{mode}.npy")


def main():
    out = []
    cases = [(f, d) for f in PLAN for d in PLAN[f][0]]
    if len(sys.argv) > 1:
        cases = [tuple(a.split("/")) for a in sys.argv[1:]]
    for fam, dn in cases:
        c = load_case(fam, dn)
        hgb = c["models"]["hgb"]
        Ftr, Xtr, X = c["Ftr"], c["Xtr"], c["Xte"]
        T = np.load(FEATS / "default" / f"{fam}__{dn}__row.npy")
        elementwise = fam in ("yj_std", "chi2")
        p_in = X.shape[1]
        src = (lambda f: f % p_in) if elementwise else None
        if elementwise:
            # an atom is a test row whose INPUT value on the lane's source
            # feature equals that of a training row whose lane value is t
            cand = np.arange(len(X))
        else:
            tr_index = {}
            for i, r in enumerate(Xtr):
                tr_index.setdefault(r.tobytes(), []).append(i)
            copies = {}
            for j, r in enumerate(X):
                ii = tr_index.get(r.tobytes())
                if ii:
                    copies[j] = ii[0]
            cand = np.array(sorted(copies), dtype=int)
            cand_tr = np.array([copies[j] for j in cand], dtype=int)
        pairs = []  # (row, f, t)
        for it in hgb._predictors:
            for p in it:
                nodes = p.nodes
                stack = [(0, np.arange(len(cand)))]
                while stack:
                    nd_i, idx = stack.pop()
                    nd = nodes[nd_i]
                    if nd["is_leaf"] or len(idx) == 0:
                        continue
                    f, t = int(nd["feature_idx"]), float(nd["num_threshold"])
                    if elementwise:
                        vals = np.unique(Xtr[Ftr[:, f] == t, src(f)])
                        at = idx[np.isin(X[cand[idx], src(f)], vals)] if len(vals) else idx[:0]
                    else:
                        at = idx[Ftr[cand_tr[idx], f] == t]
                    pairs += [(int(cand[k]), f, t) for k in at]
                    x = T[cand[idx], f]
                    stack.append((int(nd["left"]), idx[x <= t]))
                    stack.append((int(nd["right"]), idx[x > t]))
        pr = np.array([p[0] for p in pairs], dtype=int)
        pf = np.array([p[1] for p in pairs], dtype=int)
        pt = np.array([p[2] for p in pairs])
        rT = hgb._raw_predict(T)
        res = dict(case=f"{fam}/{dn}", candidates=len(cand), atom_pairs=len(pairs),
                   rows_with_atom=int(len(set(pr.tolist()))), cmps={})
        for name in CMPS + ["fit_transform"]:
            n_tr = len(Xtr)
            sel = np.ones(len(pr), bool)
            if name == "fit_transform":
                # the training rows as the model saw them (only the appended block)
                C = T.copy()
                C[-n_tr:] = Ftr
                sel = pr >= len(X) - n_tr
            else:
                C = load_cmp(fam, dn, name)
            if len(pairs) == 0:
                res["cmps"][name] = dict(pred=0.0, straddles=0, rows_changed=0)
                continue
            rows_ref = slice(len(X) - n_tr, None) if name == "fit_transform" else slice(None)
            lane_pdiff = (C[rows_ref] != T[rows_ref]).mean(axis=0)
            pred = float(0.5 * lane_pdiff[pf[sel]].sum())
            xT, xC = T[pr[sel], pf[sel]], C[pr[sel], pf[sel]]
            straddle = (xT <= pt[sel]) != (xC <= pt[sel])
            rC = hgb._raw_predict(C)
            changed = int((rT != rC).any(axis=1).sum())
            res["cmps"][name] = dict(pred=round(pred, 1), straddles=int(straddle.sum()),
                                     rows_straddling=int(len(set(pr[sel][straddle].tolist()))),
                                     rows_changed=changed)
        print(json.dumps(res), flush=True)
        out.append(res)
    (HERE / "results" / "atoms.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
