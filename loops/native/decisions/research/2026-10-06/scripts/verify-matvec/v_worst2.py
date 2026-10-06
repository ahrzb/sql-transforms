import numpy as np, math, struct, pickle, sys
D = sys.argv[1]; fx = {f["seed"]: f for f in pickle.load(open(sys.argv[2], "rb"))}
E = np.load(f"{D}/entry.npz"); sk = np.load(f"{D}/tw_SkylakeX.npz"); sb = np.load(f"{D}/tw_Sandybridge.npz")
key = sk["key"]
def ordv(x):
    i, = struct.unpack("<q", struct.pack("<d", x)); return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
ok = np.isfinite(E["S2"])
for name, a, b in (("entry-SKXrow", E["ent"], sk["row"]), ("batch-row", sk["batch"], sk["row"]), ("SNB-SKX", sb["row"], sk["row"])):
    u = np.array([abs(ordv(float(p)) - ordv(float(q))) if (p == p and q == q) else 0 for p, q in zip(a, b)], dtype=float); u[~ok] = 0
    for thr in (1e3, 1e6, 1e9):
        idx = np.where(u > thr)[0]
        has_huge = 0
        for i in idx:
            sd, iid, ri, k = key[i]; f = fx[int(sd)]; r = f["rows"][int(ri)]
            xs = [abs(float(r[n])) for n in f["names"] if r[n] is not None]
            has_huge += max(xs) >= 1e299
        print(f"{name}: lanes > {thr:.0e} ulps: {len(idx)}; with a |x_i| = 1e300 input: {has_huge}")
