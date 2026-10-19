import json, os
S = os.path.dirname(os.path.abspath(__file__))
data = [r for r in json.load(open(os.path.join(S, "probe.json"))) if "error" not in r]
def fmt(v):
    if isinstance(v, float): return f"{v:.4g}"
    return str(v)
for r in data:
    parts = []
    for x in r["rows"]:
        if x["cls"] in ("=",) : continue
        if x["probe"] not in ("+inf", "-inf", "NULL", "0", "1e308", "1e6"): continue
        f = x["feat"]
        if f != (1 if r["cls"].startswith("Select") or r["cls"] in ("RFE","RFECV","SequentialFeatureSelector","ColumnTransformer") else 0): continue
        e = x["entry"]
        if isinstance(e, dict):
            dep = x["dep"] or list(e)
            vals = ",".join(fmt(e[k]) for k in dep[:4]) + ("…" if len(dep) > 4 else "")
        else: vals = str(e)[:30]
        parts.append(f'{x["probe"]}->{x["cls"]}[{vals}]')
    if parts: print(f'{r["cfg"]:<24}', " ".join(parts))
