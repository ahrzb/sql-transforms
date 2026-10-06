import numpy as np, sys
D = sys.argv[1]; E = np.load(f"{D}/entry.npz"); sk = np.load(f"{D}/tw_SkylakeX.npz"); sb=np.load(f"{D}/tw_Sandybridge.npz")
EPS = 2.0**-52; n = E["nfeat"]; ok = np.isfinite(E["S2"])
for name, a, b in (("entry-SKXrow", E["ent"], sk["row"]), ("batch-row", sk["batch"], sk["row"]), ("entry-SNB", E["ent"], sb["row"])):
    for Sn in ("Srec", "S2"):
        S = E[Sn]
        with np.errstate(all="ignore"):
            K = np.where(a == b, 0, np.abs(a - b) / (EPS * S))
        K[~ok | ~np.isfinite(K)] = 0
        r = K / np.sqrt(n)
        i = int(np.argmax(K)); j = int(np.argmax(r))
        print(f"{name} {Sn}: maxK {K[i]:.3f} at n={n[i]} (K/sqrt n={K[i]/np.sqrt(n[i]):.3f}); max K/sqrt(n) {r[j]:.3f} at n={n[j]} K={K[j]:.3f}")
# per-n max for entry-SKX, S_rec
S = E["Srec"]; a, b = E["ent"], sk["row"]
with np.errstate(all="ignore"): K = np.where(a == b, 0, np.abs(a - b) / (EPS * S))
K[~ok | ~np.isfinite(K)] = 0
print("per n: " + " ".join(f"{m}:{K[n==m].max():.2f}" for m in sorted(set(n.tolist()))))
# dot-level: LTR vs exact, eps*S0 units
ltr, ex, S0 = E["ltr"], E["exact"], E["S0"]
g = np.isfinite(ltr) & np.isfinite(ex) & np.isfinite(S0) & (S0 > 1e-290)
e = np.abs(ltr[g] - ex[g]) / (EPS * S0[g]); print("LTR-exact max", e.max(), "p99", np.percentile(e, 99))
