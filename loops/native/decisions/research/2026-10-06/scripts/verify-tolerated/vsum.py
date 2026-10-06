import json, sys, os, collections
H = os.path.dirname(os.path.abspath(__file__))
mode = sys.argv[1]
data = json.load(open(os.path.join(H, f"vprobe_{mode}.json")))
errs = [r for r in data if "error" in r]
ok = [r for r in data if "error" not in r]
print("configs", len(data), "ok", len(ok), "errors", len(errs))
for e in errs: print("  ERR", e["cfg"], e["error"])
rows = [(r, x) for r in ok for x in r["rows"]]
print("rows", len(rows))
k = collections.Counter(x["k"] for _, x in rows)
print("kinds", dict(k))
tr = [(r, x) for r, x in rows if x["k"] in ("c", "answer")]
print("twin raises", len(tr), " entry raises there (c)", k["c"])
ans = [(r, x) for r, x in tr if x["k"] == "answer"]
allfin = [(r, x) for r, x in ans if x["all_finite"]]
print("entry answers all-finite lanes on", len(allfin))
# dep-based classes
cl = collections.Counter()
for r, x in ans:
    dt = x["dep_t"]
    if not dt:
        c = "b0" if x["all_finite"] else "a0"
    elif x["dep_finite"]:
        c = "b"
    elif not x["dep_any_finite"]:
        c = "a"
    else:
        c = "ab"
    x["cls2"] = c
    cl[c] += 1
print("classes (twin-dep):", dict(cl))
# ab rows that are not all-finite
print("ab rows all-finite?", collections.Counter(x["all_finite"] for r, x in ans if x["cls2"] == "ab"))
print("b rows all-finite?", collections.Counter(x["all_finite"] for r, x in ans if x["cls2"] == "b"))
# dep_e vs dep_t disagreement
dis = sum(1 for r, x in rows if x["dep_e"] != x["dep_t"])
print("rows where entry-dep != twin-dep:", dis)
cfg_b = {r["cfg"] for r, x in ans if x["cls2"] in ("b", "ab")}
cls_b = {r["cls"] for r, x in ans if x["cls2"] in ("b", "ab")}
print("configs with b/ab:", len(cfg_b), "classes:", len(cls_b))
cfg_any = {r["cfg"] for r, x in ans if x["cls2"] in ("b", "ab", "b0")}
print("configs with b/ab/b0:", len(cfg_any), "classes", len({r['cls'] for r, x in ans if x['cls2'] in ('b','ab','b0')}))
nullb = sorted({r["cfg"] for r, x in ans if x["probe"] == "NULL" and x["cls2"] in ("b", "ab", "b0")})
print("NULL -> b/ab/b0 configs:", len(nullb))
nullf = sorted({r["cfg"] for r, x in ans if x["probe"] == "NULL" and x["all_finite"]})
print("NULL -> all-finite configs:", len(nullf))
# ref comparison
rc = collections.Counter((x["cls2"] in ("b", "ab", "b0"), x["ref_same"]) for r, x in ans)
print("ref_same by (finite-class, ref_same):", dict(rc))
rc2 = collections.Counter(x["ref_same"] for r, x in allfin)
print("ref_same among all-finite answers:", dict(rc2))
diff = collections.Counter((r["cls"], x["probe"]) for r, x in ans if x["cls2"] in ("b", "ab", "b0") and x["ref_same"] is False)
print("finite-class rows != sklearn(assume_finite):", dict(diff))
still = collections.Counter((r["cls"], x["probe"], x["ref"][:60] if isinstance(x["ref"], str) else "") for r, x in ans if x["cls2"] in ("b", "ab", "b0") and x["ref_same"] is None)
print("finite-class rows where sklearn(assume_finite) still raises:", dict(still))
# DIFF / REVERSE detail
for r, x in rows:
    if x["k"] in ("DIFF", "REVERSE"):
        print("  ", x["k"], r["cfg"], x["feat"], x["probe"], str(x["twin"])[:100], str(x["entry"])[:100])
# per class: b/ab rows
pc = collections.defaultdict(collections.Counter)
for r, x in ans: pc[r["cls"]][x["cls2"]] += 1
for c, v in pc.items(): print(f"  {c:<26} {dict(v)}")
