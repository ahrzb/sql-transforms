"""Twin outputs for pickled fixtures, under whatever numpy dispatch is active.
usage: vtwin.py IN.pkl OUT.pkl"""
import sys, pickle, math, warnings
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
data = pickle.load(open(sys.argv[1], "rb"))
res = []
for d in data:
    out = []
    for r in d["rows"]:
        iid = r["__iid"]
        if iid is None:
            out.append(None); continue
        est = d["inst"][iid]
        vals = [math.nan if r[f] is None else float(r[f]) for f in d["names"]]
        try:
            out.append(np.asarray(est.transform(np.array([vals])), dtype=float)[0])
        except ValueError:
            out.append(None)
    res.append(out)
pickle.dump(res, open(sys.argv[2], "wb"))
