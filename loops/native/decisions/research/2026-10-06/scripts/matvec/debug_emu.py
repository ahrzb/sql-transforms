import pickle, sys, warnings, math
sys.path.insert(0, sys.argv[1])
import numpy as np
from emul import numpy_row_matvec
from collections import Counter
warnings.simplefilter("ignore")
fixtures = pickle.load(open(sys.argv[2], "rb"))
bad = Counter(); tot = Counter(); ex = []
for fx in fixtures:
    for iid, est in fx["instances"].items():
        C = est.components_; Cl = C.tolist()
        for r in fx["rows"]:
            if r["__iid"] != iid: continue
            vals = [float("nan") if r[n] is None else float(r[n]) for n in fx["names"]]
            if any(math.isnan(v) for v in vals): continue
            y = (np.array([vals]) @ C.T)[0].tolist()
            e = numpy_row_matvec(vals, Cl, "SkylakeX")
            big = any(abs(v) >= 1e300 for v in vals)
            for k in range(len(Cl)):
                key = (len(Cl) == 1, bool(C.flags.c_contiguous))
                tot[key] += 1
                if not (e[k] == y[k] or (math.isnan(e[k]) and math.isnan(y[k]))):
                    bad[key] += 1
                    if len(ex) < 6: ex.append((len(vals), len(Cl), k, y[k], e[k], vals))
print("(k==1, components_ C-contiguous): bad/total", {k: f"{bad[k]}/{tot[k]}" for k in tot})
for t in ex: print(t[:5])
