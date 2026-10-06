import json, os, collections, re
S = os.path.dirname(os.path.abspath(__file__))
data = [r for r in json.load(open(os.path.join(S, "probe.json"))) if "error" not in r]
def reason(m):
    for pat, lab in [("infinity", "infinity"), ("contains NaN", "NaN"), ("unknown categories", "unknown category"),
                     ("strictly positive", "box-cox x<=0"), ("interpolation range", "isotonic out of range"),
                     ("have missing values in transform but have no missing values in fit", "MissingIndicator new missing"),
                     ("X contains values beyond the limits of the knots", "spline beyond knots"),
                     ("greater than|less than|out of bounds|outside", "range")]:
        if re.search(pat, m): return lab
    return m[-90:]
agg = collections.defaultdict(collections.Counter)
for r in data:
    for x in r["rows"]:
        if isinstance(x["twin"], str) and x["twin"].startswith("raise"):
            agg[r["cls"]][(x["probe"], reason(x["twin"]))] += 1
for k, v in agg.items():
    byp = collections.defaultdict(set)
    for (p, why), n in v.items(): byp[why].add(p)
    print(f"{k:<26}", "; ".join(f"{w}: {','.join(sorted(ps))}" for w, ps in byp.items()))
