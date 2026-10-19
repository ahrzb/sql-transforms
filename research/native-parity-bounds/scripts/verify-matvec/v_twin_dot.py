import pickle, sys, math, numpy as np
D = sys.argv[1]; fx = {f["seed"]: f for f in pickle.load(open(sys.argv[2], "rb"))}
E = np.load(f"{D}/entry.npz"); key = np.load(f"{D}/tw_SkylakeX.npz")["key"]; EPS = 2.0**-52
dot = np.empty(len(key))
cache = {}
for j, (sd, iid, ri, k) in enumerate(key):
    f = fx[int(sd)]; est = f["instances"][int(iid)]; r = f["rows"][int(ri)]
    ck = (int(sd), int(iid), int(ri))
    if ck not in cache:
        x = np.array([float(r[n]) for n in f["names"]]); cache = {ck: (x[None, :] @ est.components_.T)[0]}
    dot[j] = cache[ck][int(k)]
ex, S0, ltr = E["exact"], E["S0"], E["ltr"]
g = np.isfinite(ex) & np.isfinite(S0) & (S0 > 1e-290) & np.isfinite(dot) & np.isfinite(ltr)
for lim in (200, 1000):
    m = g & (E["seeds"] < lim)
    e1 = np.abs(dot[m] - ex[m]) / (EPS * S0[m]); e2 = np.abs(ltr[m] - ex[m]) / (EPS * S0[m])
    print(f"seeds<{lim}: SKX gemv-exact max {e1.max():.3f}; LTR-exact max {e2.max():.3f}; lanes {m.sum()}")
