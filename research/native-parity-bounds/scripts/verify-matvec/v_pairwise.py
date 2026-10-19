"""Alternative entry order at the same cost as LTR: a balanced pairwise tree
over the n products (no FMA). K against the SkylakeX row twin (S2) and the
dot's error vs exact, on own synthetic data."""
import pickle, sys, math, numpy as np
V = sys.argv[1]; EPS = 2.0**-52
data = pickle.load(open(f"{V}/syn.pkl", "rb")); T = pickle.load(open(f"{V}/syn_SkylakeX.pkl", "rb")); TS = pickle.load(open(f"{V}/syn_Sandybridge.pkl", "rb"))
def pairwise(P):  # P: (..., n) products, sum along last axis as a balanced tree
    while P.shape[-1] > 1:
        n = P.shape[-1]; h = n // 2
        Q = P[..., :h] + P[..., h:2 * h]
        P = np.concatenate([Q, P[..., 2 * h:]], axis=-1) if n % 2 else Q
    return P[..., 0]
print(f"{'data':6s} {'n':>5s} {'LTR K(S2) vs SKX':>17s} {'pairwise K(S2) vs SKX':>22s} {'pairwise K vs SNB':>18s} {'LTR K vs SNB':>13s}")
for (name, n), d in data.items():
    if name == "zmean": continue
    est, X = d["est"], d["X"]; C = est.components_
    s = np.sqrt(est.explained_variance_); s[s < np.finfo(float).eps] = np.finfo(float).eps
    mc = T[(name, n)]["mc"]
    P = X[:, None, :] * C[None, :, :]
    pw = (pairwise(P) - mc) / s
    acc = X[:, :1] * C[:, 0]
    for i in range(1, n): acc = acc + X[:, i:i + 1] * C[:, i]
    lt = (acc - mc) / s
    S2 = ((np.abs(X) @ np.abs(C).T) + np.abs(est.mean_) @ np.abs(C).T) / s
    tw = T[(name, n)]["row"]; sb = TS[(name, n)]["row"]
    K = lambda a, b: np.max(np.where(a == b, 0, np.abs(a - b) / (EPS * S2)))
    print(f"{name:6s} {n:5d} {K(lt, tw):17.2f} {K(pw, tw):22.2f} {K(pw, sb):18.2f} {K(lt, sb):13.2f}")
