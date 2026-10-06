import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/tolerated")
import shim  # noqa
import json, collections, warnings
from sql_transform.native.catalog_test import FIXTURES, _runs
warnings.simplefilter("ignore")
mode = sys.argv[1]
data = {r["cfg"]: r for r in json.load(open(f"/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/verify-tolerated/vprobe_{mode}.json")) if "error" not in r}
res = collections.defaultdict(lambda: [0, 0, 0, collections.Counter(), collections.Counter()])
for cls, fs in FIXTURES.items():
    for j, f in enumerate(fs):
        lab = f"{cls.__name__}[{j}]"
        if lab not in data: continue
        est = f()
        t = est.__sklearn_tags__()
        an, nv = t.input_tags.allow_nan, getattr(t, "no_validation", False)
        # leaf-tag variant: allow_nan only if every run allows it; no_validation if all runs
        runs = _runs(est)
        an_l = all(r.__sklearn_tags__().input_tags.allow_nan for r in runs)
        nv_l = all(getattr(r.__sklearn_tags__(), "no_validation", False) for r in runs)
        for x in data[lab]["rows"]:
            tr = x["k"] in ("c", "answer")
            p = x["probe"]
            isinf = p in ("+inf", "-inf"); isnan = p in ("NaN", "NULL")
            gs = {"g1": isinf, "g2": isinf or (isnan and not an), "g3": (not nv) and (isinf or (isnan and not an)),
                  "g4_leaf": (not nv_l) and (isinf or (isnan and not an_l))}
            for g, v in gs.items():
                r = res[g]
                if tr: r[0] += 1
                if tr and v: r[1] += 1
                if tr and not v: r[3][cls.__name__] += 1
                if v and not tr: r[4][cls.__name__] += 1
for g, (t, hit, _, miss, over) in res.items():
    print(f"{g}: twin raises {t}, catches {hit}, misses {sum(miss.values())}, over-raises {sum(over.values())}")
    print("    misses", dict(miss)); print("    over  ", dict(over))
