"""V9: calibrate the continuous flip model c(t)*E|delta| by injecting KNOWN
random perturbations (delta = d * sign) large enough to observe, and compare
observed DT/RF leaf changes with (a) the report's k=16 estimator and (b) my
window estimator; then extrapolate linearly to the measured N-vs-T E|delta|."""
import sys, warnings
import numpy as np
R = "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/inference"
sys.path.insert(0, R)
warnings.filterwarnings("ignore")
from common import FEATS, load_case  # noqa
from analyze import sk_tree_nodes, expected_straddles  # noqa  (the report's estimator)


def window_nodes(tree, X32, wfrac=1e-3):
    t = tree; res = []; stack = [(0, np.arange(len(X32)))]
    sd = X32.astype(np.float64).std(0)
    while stack:
        node, idx = stack.pop()
        if t.children_left[node] == -1 or len(idx) == 0:
            continue
        f, thr = t.feature[node], t.threshold[node]
        x = X32[idx, f].astype(np.float64)
        w = wfrac * sd[f]
        res.append((f, np.sum(np.abs(x - thr) < w) / (2 * w)))
        gl = X32[idx, f] <= thr
        stack.append((t.children_left[node], idx[gl])); stack.append((t.children_right[node], idx[~gl]))
    return res


def main(fam, dn):
    c = load_case(fam, dn); m = c["models"]; m["rf"].n_jobs = 4
    T = np.load(FEATS / "default" / f"{fam}__{dn}__row.npy")
    N = np.load(FEATS / "native" / f"{fam}__{dn}__N.npy")
    nn = len(T) - len(c["Xtr"]); T = T[:nn]; N = N[:nn]   # new rows only
    T32 = T.astype(np.float32)
    dt_k = sk_tree_nodes(m["dtree"].tree_, T32)
    rf_k = [sk_tree_nodes(e.tree_, T32) for e in m["rf"].estimators_]
    dt_w = window_nodes(m["dtree"].tree_, T32)
    rf_w = [window_nodes(e.tree_, T32) for e in m["rf"].estimators_]
    rng = np.random.default_rng(99)
    sd = T.std(0)
    leafT = m["dtree"].apply(T32); rfT = m["rf"].apply(T32)
    print(f"{fam}/{dn}: new rows {nn}, DT splits {len(dt_k)}")
    for rel in (1e-7, 1e-6, 1e-5):
        D = rel * sd[None, :] * rng.choice([-1.0, 1.0], size=T.shape)
        C = T + D; C32 = C.astype(np.float32)
        obs_dt = int((m["dtree"].apply(C32) != leafT).sum())
        obs_rf = int((m["rf"].apply(C32) != rfT).sum())  # tree-row leaf changes, summed over trees
        ad = np.abs(C32.astype(np.float64) - T32.astype(np.float64))
        e_dt_k = expected_straddles(dt_k, ad); e_rf_k = sum(expected_straddles(n, ad) for n in rf_k)
        e_dt_w = sum(cc * ad[:, f].mean() for f, cc in dt_w); e_rf_w = sum(cc * ad[:, f].mean() for nodes in rf_w for f, cc in nodes)
        print(f"  rel {rel:.0e}: DT obs {obs_dt} k16-est {e_dt_k:.1f} window-est {e_dt_w:.1f} | RF(100 trees, tree-row changes) obs {obs_rf} k16-est {e_rf_k:.1f} window-est {e_rf_w:.1f}")
    adN = np.abs(N - T)
    e_dt = sum(cc * adN[:, f].mean() for f, cc in dt_w); e_rf = sum(cc * adN[:, f].mean() for nodes in rf_w for f, cc in nodes)
    print(f"  N-vs-T (float64 |delta|, window est): DT {e_dt:.2e}  RF(100 trees) {e_rf:.2e}  per row per tree {e_rf / nn / 100:.1e}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        main(*a.split("/"))
