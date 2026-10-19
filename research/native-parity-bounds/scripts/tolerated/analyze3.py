import json, os, collections
S = os.path.dirname(os.path.abspath(__file__))
data = [r for r in json.load(open(os.path.join(S, "probe.json"))) if "error" not in r]
c = collections.Counter(); per = collections.defaultdict(collections.Counter)
ex = {}
for r in data:
    for x in r["rows"]:
        if x["cls"] in ("=", "DIFF", "REVERSE", "c"): continue
        k = {True: "entry == sklearn(assume_finite)", False: "entry != sklearn(assume_finite)", None: "sklearn(assume_finite) still raises"}[x.get("ref_same")]
        c[(x["cls"] in ("b","ab","b0"), k)] += 1
        if x["cls"] in ("b","ab","b0") and x.get("ref_same") is False:
            per[r["cls"]][x["probe"]] += 1
            ex.setdefault((r["cls"], x["probe"]), (r["cfg"], x["feat"], x["entry"], x["ref"]))
for k, v in sorted(c.items(), key=str): print(k, v)
print("finite answers that differ from sklearn-without-validation, by class/probe:")
for k, v in per.items(): print(" ", k, dict(v))
for k, v in list(ex.items()):
    print("  ex", k, v[0], "feat", v[1], "entry", str(v[2])[:110], "ref", str(v[3])[:110])
