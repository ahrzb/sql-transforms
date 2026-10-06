"""Why do the matvec record's lane counts (3,855 at seeds 0-39; 23,050 at
0-199) not match the 4,618 / 25,208 everyone re-measured? Try filters."""
import pickle
import warnings

import numpy as np

P = "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/matvec/fixtures200.pkl"
fx = pickle.load(open(P, "rb"))
warnings.simplefilter("ignore")
rec = []  # (seed, any_huge, any_tiny_edge, lane_finite, n_lanes)
for f in fx:
    for r in f["rows"]:
        est = f["instances"].get(r["__iid"])
        if est is None:
            continue
        vals = [float("nan") if r[n] is None else float(r[n]) for n in f["names"]]
        try:
            out = est.transform([vals])[0]
        except ValueError:
            continue
        huge = any(abs(v) >= 1e300 for v in vals)
        big = any(abs(v) >= 1e100 for v in vals)
        for y in out:
            rec.append((f["seed"], huge, big, bool(np.isfinite(y)), abs(y)))
a = np.array([(s, h, b, fi) for s, h, b, fi, _ in rec], dtype=float)
mag = np.array([m for *_, m in rec])
for upto, want in [(40, 3855), (200, 23050)]:
    m = a[:, 0] < upto
    print(f"seeds<{upto} (record {want}): all {int(m.sum())}; "
          f"no |x|>=1e300 row {int((m & (a[:,1]==0)).sum())}; "
          f"no |x|>=1e100 row {int((m & (a[:,2]==0)).sum())}; "
          f"finite lane {int((m & (a[:,3]==1)).sum())}; "
          f"|y|<1e283 {int((m & (mag < 2.6e283)).sum())}; "
          f"|y|<1e100 {int((m & (mag < 1e100)).sum())}")
