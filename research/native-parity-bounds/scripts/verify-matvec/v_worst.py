import numpy as np, math, struct, pickle, sys
D = sys.argv[1]; fx = {f["seed"]: f for f in pickle.load(open(sys.argv[2], "rb"))}
E = np.load(f"{D}/entry.npz"); sk = np.load(f"{D}/tw_SkylakeX.npz"); sb = np.load(f"{D}/tw_Sandybridge.npz")
key = sk["key"]; EPS = 2.0**-52
def ordv(x):
    i, = struct.unpack("<q", struct.pack("<d", x)); return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
def ul(a, b): return abs(ordv(a) - ordv(b))
ent, row, bat = E["ent"], sk["row"], sk["batch"]
S2, Srec, s = E["S2"], E["Srec"], E["s"]
ok = np.isfinite(S2)
for name, a, b in (("entry-SKXrow", ent, row), ("SKXbatch-SKXrow", bat, row), ("SNB-SKX", sb["row"], row)):
    u = np.array([ul(float(p), float(q)) if (p == p and q == q) else 0 for p, q in zip(a, b)], dtype=float)
    u[~ok] = 0
    top = np.argsort(-u)[:4]
    print(name)
    for i in top:
        sd, iid, ri, k = key[i]; est = fx[int(sd)]["instances"][int(iid)]
        ev = est.explained_variance_; evr = ev[int(k)] / ev[0]
        print(f"  ulps {u[i]:.3g} a {a[i]:.6g} b {b[i]:.6g} s {s[i]:.3g} ev_k/ev_0 {evr:.2e} k {k}/{len(ev)} n {est.n_features_in_} K_S2 {abs(a[i]-b[i])/(EPS*S2[i]):.2f} K_Srec {abs(a[i]-b[i])/(EPS*Srec[i]):.2f}")
    # how many lanes with >1e6 ulps have tiny s / ev ratio
    big = np.where(u > 1e6)[0]
    rat = []
    for i in big:
        sd, iid, ri, k = key[i]; ev = fx[int(sd)]["instances"][int(iid)].explained_variance_; rat.append(ev[int(k)] / ev[0])
    rat = np.array(rat)
    print(f"  lanes >1e6 ulps: {len(big)}; of them ev_k/ev_0 < 1e-20: {(rat < 1e-20).sum()}; s==eps: {(s[big] == np.finfo(float).eps).sum()}")
