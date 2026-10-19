import math, pickle, struct, numpy as np
from fractions import Fraction as F
EPS = 2.0 ** -52
L = pickle.load(open("v1_lanes.pkl", "rb"))
def od(v):
    (i,) = struct.unpack("<q", struct.pack("<d", v)); return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
def ul(a, b): return abs(od(a) - od(b))
fin = lambda *v: all(math.isfinite(t) for t in v)
def pair(name, fa, fb, sk="S"):
    us, Ks, dif, worst = [], [], 0, None
    for l in L:
        a, b, s = fa(l), fb(l), l[sk]
        if not fin(a, b, s) or s <= 0: continue
        u = ul(a, b); us.append(u); dif += a != b
        K = abs(a - b) / (EPS * s); Ks.append(K)
        if worst is None or u > worst[0]: worst = (u, K, l["n"], a, b, s)
    us = np.array(us, float); Ks = np.array(Ks)
    print(f"{name}: lanes {len(us)}, differ {dif}, max ulps {us.max():.0f}, p99 ulps {np.percentile(us,99):.0f}, p99 K {np.percentile(Ks,99):.3f}, max K {Ks.max():.3f}; worst-ulp lane K={worst[1]:.3f} n={worst[2]} a={worst[3]!r} b={worst[4]!r}")
    return Ks
pair("entry vs twin[SkylakeX]", lambda l: l["ent"], lambda l: l["tw"])
pair("entry vs twin[Haswell]", lambda l: l["ent"], lambda l: l["hw"])
# entry vs SB twin: the entry uses the default-kernel M (as translation on SkylakeX machine), and alternatively SB's M
pair("entry(M_SkX) vs twin[Sandybridge] S", lambda l: l["ent"], lambda l: l["sb"])
pair("entry(M_SkX) vs twin[Sandybridge] S_full", lambda l: l["ent"], lambda l: l["sb"], "Sf")
def ent_sb(l):
    # entry with Sandybridge's M
    return None
pair("twin[SkX] vs twin[SB] S_full", lambda l: l["tw"], lambda l: l["sb"], "Sf")
pair("twin[SkX] vs twin[SB] S", lambda l: l["tw"], lambda l: l["sb"], "S")
pair("twin[SkX] vs twin[HW] S_full", lambda l: l["tw"], lambda l: l["hw"], "Sf")
pair("twin row vs twin batched S", lambda l: l["tw"], lambda l: l["bt"])
pair("twin run1 vs run2", lambda l: l["tw"], lambda l: l["again"])
# vs exact
def vsx(name, fa, key="ex", sk="Sf"):
    us, Ks = [], []; worst = None; nonex = 0
    for l in L:
        a = fa(l)
        if not math.isfinite(a): continue
        try: r = float(l[key])
        except OverflowError: continue
        if not math.isfinite(r): continue
        u = ul(a, r); us.append(u); nonex += u != 0
        K = float(abs(F(a) - l[key])) / (EPS * l[sk]) if l[sk] > 0 else 0; Ks.append(K)
        if worst is None or u > worst[0]: worst = (u, l)
    us = np.array(us, float)
    print(f"{name} vs correctly rounded ({key}): lanes {len(us)}, not exact {nonex}, max ulps {us.max():.3g}, p99 {np.percentile(us,99):.0f}, p99.9 {np.percentile(us,99.9):.0f}, max K({sk}) {max(Ks):.3f}")
    return worst
w = vsx("twin", lambda l: l["tw"])
l = w[1]; print("  worst twin lane:", dict(n=l["n"], tw=l["tw"], ent=l["ent"], ex=float(l["ex"]), exM=float(l["exM"]), M=l["M"], S=l["S"]))
vsx("entry", lambda l: l["ent"])
vsx("twin", lambda l: l["tw"], key="exM", sk="S")
vsx("entry", lambda l: l["ent"], key="exM", sk="S")
