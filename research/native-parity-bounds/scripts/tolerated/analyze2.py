import json, sys, os, collections
S = os.path.dirname(os.path.abspath(__file__))
data = [r for r in json.load(open(os.path.join(S, "probe.json"))) if "error" not in r]
# 1. classes affected
by_cls = collections.defaultdict(lambda: collections.Counter())
cfg_b = collections.Counter(); cfg_tot = collections.Counter()
for r in data:
    c = collections.Counter(x["cls"] for x in r["rows"])
    by_cls[r["cls"]].update(c)
    cfg_tot[r["cls"]] += 1
    if c["b"] or c["ab"] or c["b0"]:
        cfg_b[r["cls"]] += 1
print("class: configs with a silently-finite answer (b/ab/b0) / configs")
for k in by_cls:
    print(f"  {k:<26} {cfg_b[k]}/{cfg_tot[k]}  {dict(by_cls[k])}")
print("classes with any b/ab/b0:", sum(1 for k in by_cls if cfg_b[k]), "of", len(by_cls))
print("configs with any b/ab/b0:", sum(cfg_b.values()), "of", sum(cfg_tot.values()))
# strictly b (dependent lane finite)
cfg_bb = collections.Counter()
for r in data:
    if any(x["cls"] in ("b","ab") for x in r["rows"]): cfg_bb[r["cls"]] += 1
print("configs with b/ab (a lane that reads the bad input is finite):", sum(cfg_bb.values()), "classes:", len(cfg_bb), sorted(cfg_bb))
# by probe kind
pk = collections.defaultdict(collections.Counter)
for r in data:
    for x in r["rows"]:
        pk[x["probe"]][x["cls"]] += 1
print("by probe:")
for p, c in pk.items(): print(f"  {p:<7} {dict(c)}")
# NULL specifically: classes where NULL -> b/ab/b0
nullb = sorted({r["cfg"] for r in data for x in r["rows"] if x["probe"]=="NULL" and x["cls"] in ("b","ab","b0")})
print("NULL -> finite where twin raises:", len(nullb), nullb)
# guards
print("guards (g1 inf-any, g2 inf or NaN&!allow_nan, g3 = g2 unless no_validation):")
for gi, name in enumerate(["g1","g2","g3"]):
    miss = collections.Counter(); over = collections.Counter(); hit = 0; tw_r = 0
    for r in data:
        for x in r["rows"]:
            tr = isinstance(x["twin"], str) and x["twin"].startswith("raise")
            g = x["g"][gi]
            tw_r += tr
            if tr and not g: miss[r["cls"]] += 1
            if g and not tr: over[r["cls"]] += 1
            if g and tr: hit += 1
    print(f"  {name}: twin raises {tw_r}, guard catches {hit}, misses {sum(miss.values())} {dict(miss)}")
    print(f"      over-raises (forbidden) {sum(over.values())} {dict(over)}")
