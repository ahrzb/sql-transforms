import pickle, sys, math, struct, warnings
import numpy as np
D = sys.argv[1]; fx = pickle.load(open(sys.argv[2], "rb"))
EPS = 2.0**-52
T = {ct: np.load(f"{D}/tw_{ct}.npz") for ct in ["SkylakeX", "Haswell", "Sandybridge", "Nehalem", "Prescott"]}
key = T["SkylakeX"]["key"]
def ordv(x):
    i, = struct.unpack("<q", struct.pack("<d", x)); return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
def ulps(a, b):
    if math.isnan(a) or math.isnan(b): return 0 if (math.isnan(a) and math.isnan(b)) else 2**64
    return abs(ordv(a) - ordv(b))
# own entry, scales
mcS = T["SkylakeX"]["mc1"]
ent, Srec, S2, S0a, nfeat, seeds, ltr_dot, exact_dot, scl = [], [], [], [], [], [], [], [], []
byseed = {f["seed"]: f for f in fx}
for j, (seed, iid, ri, k) in enumerate(key):
    f = byseed[int(seed)]; est = f["instances"][int(iid)]; r = f["rows"][int(ri)]
    x = [float("nan") if r[nm] is None else float(r[nm]) for nm in f["names"]]
    c = est.components_[int(k)].tolist(); m = est.mean_.tolist()
    s = math.sqrt(float(est.explained_variance_[int(k)]))
    if s < np.finfo(float).eps: s = np.finfo(float).eps
    acc = x[0] * c[0]
    for i in range(1, len(x)): acc = acc + x[i] * c[i]
    e = (acc - float(mcS[j])) / s
    s0 = sum(abs(a * b) for a, b in zip(x, c))
    sm = sum(abs(a * b) for a, b in zip(m, c))
    ent.append(e); Srec.append((s0 + abs(float(mcS[j]))) / s); S2.append((s0 + sm) / s); S0a.append(s0)
    nfeat.append(len(x)); seeds.append(int(seed)); ltr_dot.append(acc); scl.append(s)
    parts = []
    for a, b in zip(x, c):
        p = a * b; parts.append(p)
        if math.isfinite(p): parts.append(math.fma(a, b, -p))
    try: exact_dot.append(math.fsum(parts))
    except Exception: exact_dot.append(float("nan"))
ent = np.array(ent); Srec = np.array(Srec); S2 = np.array(S2); seeds = np.array(seeds); nfeat = np.array(nfeat)
np.savez(f"{D}/entry.npz", ent=ent, Srec=Srec, S2=S2, S0=np.array(S0a), seeds=seeds, nfeat=nfeat, ltr=np.array(ltr_dot), exact=np.array(exact_dot), s=np.array(scl))
print("lanes", len(ent), "finite Srec", np.isfinite(Srec).sum(), "finite S2", np.isfinite(S2).sum(), "max n", nfeat.max())

def stats(a, b, S, mask):
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    idx = np.where(mask & ~same)[0]
    u = max([ulps(float(a[i]), float(b[i])) for i in idx], default=0)
    with np.errstate(all="ignore"):
        K = np.abs(a[idx] - b[idx]) / (EPS * S[idx])
    Kf = K[np.isfinite(K)]
    # p99 over all masked lanes incl. equal ones (K=0)
    Kall = np.zeros(mask.sum()); 
    pos = np.searchsorted(np.where(mask)[0], idx); Kall[pos] = np.where(np.isfinite(K), K, 0)
    nonfinK = (~np.isfinite(K)).sum()
    return f"differ {len(idx):6d} maxulps {u:.4g} p99K {np.percentile(Kall,99):.3f} maxK {Kf.max() if len(Kf) else 0:.3f} nonfiniteK {nonfinK}"
row = {ct: T[ct]["row"] for ct in T}; bat = {ct: T[ct]["batch"] for ct in T}
pairs = [("entry vs SKX row", ent, row["SkylakeX"]), ("SKX batch vs SKX row", bat["SkylakeX"], row["SkylakeX"]),
         ("HSW row vs SKX row", row["Haswell"], row["SkylakeX"]), ("SNB row vs SKX row", row["Sandybridge"], row["SkylakeX"]),
         ("NHM row vs SNB row", row["Nehalem"], row["Sandybridge"]), ("PRS row vs SNB row", row["Prescott"], row["Sandybridge"]),
         ("entry vs SNB row", ent, row["Sandybridge"]), ("entry vs HSW row", ent, row["Haswell"]),
         ("SNB batch vs SKX row", bat["Sandybridge"], row["SkylakeX"]), ("entry vs SKX batch", ent, bat["SkylakeX"]),
         ("HSW batch vs SKX batch", bat["Haswell"], bat["SkylakeX"])]
for upto in (40, 200, 1000):
    for Sname, S in (("Srec", Srec), ("S2", S2)):
        mask = (seeds < upto) & np.isfinite(S)
        print(f"\n== seeds <{upto} scale {Sname} lanes {mask.sum()} (all {(seeds<upto).sum()})")
        for name, a, b in pairs:
            print(f"  {name:24s} {stats(a, b, S, mask)}")
# mc in-process: sklearn spelling vs C @ mean_
for ct in T:
    d = T[ct]["mc1"] != T[ct]["mc2"]
    print(ct, "mc (1,n)@C.T vs C@mean differ lanes:", int(d.sum()), " mc vs SKX mc differ:", int((T[ct]["mc1"] != mcS).sum()))
