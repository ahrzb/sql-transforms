import math, pickle, struct, numpy as np
from fractions import Fraction as F
EPS = 2.0 ** -52
d = pickle.load(open("v1_default.pkl", "rb"))
alt = {k: pickle.load(open(f"v1_{k}.pkl", "rb")) for k in ["Sandybridge", "Haswell", "default_again"]}
def od(v):
    (i,) = struct.unpack("<q", struct.pack("<d", v)); return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
def ul(a, b): return abs(od(a) - od(b))
L = []  # per lane dict
for r, ((key, x), tw, bt, M) in enumerate(zip(d["recs"], d["row"], d["bat"], d["M"])):
    est = d["ests"][key]; C = est.components_; k, n = C.shape
    sc = np.sqrt(est.explained_variance_); sc[sc < np.finfo(float).eps] = np.finfo(float).eps
    m = [float(v) for v in est.mean_]
    for j in range(k):
        c = [float(v) for v in C[j]]
        acc = x[0] * c[0]
        for i in range(1, n): acc = acc + x[i] * c[i]
        A = math.fsum(abs(x[i] * c[i]) for i in range(n)); B = math.fsum(abs(m[i] * c[i]) for i in range(n))
        Mj = float(M[j]); s = float(sc[j])
        ent = (acc - Mj) / s
        dx = sum(F(x[i]) * F(c[i]) for i in range(n)); dm = sum(F(m[i]) * F(c[i]) for i in range(n))
        ex = (dx - dm) / F(s)            # exact with exact mean projection
        exM = (dx - F(Mj)) / F(s)        # exact given the twin's rounded M
        L.append(dict(r=r, j=j, n=n, tw=float(tw[j]), bt=float(bt[j]), ent=ent, S=(A + abs(Mj)) / s, Sf=(A + B) / s,
                      ex=ex, exM=exM, M=Mj, sb=float(alt["Sandybridge"]["row"][r][j]), hw=float(alt["Haswell"]["row"][r][j]),
                      Msb=float(alt["Sandybridge"]["M"][r][j]), again=float(alt["default_again"]["row"][r][j])))
pickle.dump(L, open("v1_lanes.pkl", "wb"))
print("lanes total", len(L))
