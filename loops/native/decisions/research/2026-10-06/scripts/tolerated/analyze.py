import json, sys, os, collections
S = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(S, sys.argv[1] if len(sys.argv) > 1 else "probe.json")))
errs = [r for r in data if "error" in r]
ok = [r for r in data if "error" not in r]
print("configs", len(data), "served", len(ok), "errors", len(errs))
for r in errs: print("  ERR", r["cfg"], r["error"][:150])
tot = collections.Counter()
for r in ok:
    c = collections.Counter(x["cls"] for x in r["rows"])
    tot.update(c)
    # per probe: classes where twin raises
    per = collections.defaultdict(set)
    for x in r["rows"]:
        if x["cls"] not in ("=",):
            per[x["probe"]].add(f'{x["cls"]}@{x["feat"]}')
    s = "; ".join(f"{p}:{','.join(sorted(v))}" for p, v in per.items())
    print(f'{r["cfg"]:<28} an={int(r["allow_nan"])} nv={int(r["no_validation"])} {dict(c)} | {s}')
print("TOTAL", dict(tot))
