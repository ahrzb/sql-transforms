"""Census of the parity test's rows (seeds 0..SEEDS-1, every FIXTURES config):
NULL-id rows, rows where the twin raises, rows the twin answers, and of those
the rows holding a finite extreme (-0.0, ±1e300, 1e-300, 5e-324). Twin only; no
confit build. Read-only."""
import math
import os
import warnings
from collections import defaultdict

import sql_transform.native.catalog_test as ct

SEEDS = int(os.environ.get("CENSUS_SEEDS", "8"))
EXTREMES = {1e-300, 1e300, 5e-324}
warnings.simplefilter("ignore")


def extreme(v):
    if not isinstance(v, float) or math.isnan(v) or math.isinf(v):
        return False
    return abs(v) in EXTREMES or (v == 0.0 and math.copysign(1.0, v) < 0)


tot = defaultdict(int)
by_width = defaultdict(lambda: defaultdict(int))
dead_guard = 0
steps = 0
for cls, makes in ct.FIXTURES.items():
    for j, make in enumerate(makes):
        for seed in range(SEEDS):
            try:
                step = ct._step(make, seed, j)
            except Exception:
                continue
            steps += 1
            rows = ct._rows(step, seed, getattr(make, "positive", False)).to_pylist()
            w = len(step.takes)
            band = "01-04" if w <= 4 else ("05-16" if w <= 16 else "17-32")
            real = 0
            for row in rows:
                iid = row.pop("__iid")
                vals = list(row.values())
                if iid is None:
                    k = "null_id"
                else:
                    try:
                        step(iid, *vals)
                        k = "answered"
                        real += 1
                        if any(extreme(v) for v in vals):
                            tot["answered_extreme"] += 1
                            by_width[band]["extreme"] += 1
                    except Exception:
                        k = "raised"
                tot[k] += 1
                by_width[band][k] += 1
            if real == 0:
                dead_guard += 1

print("steps", steps, "rows", tot["null_id"] + tot["raised"] + tot["answered"])
for k in ("null_id", "raised", "answered", "answered_extreme"):
    print(f"  {k}: {tot[k]}")
print("steps whose 'compared' rows are all NULL-id rows:", dead_guard)
for band, d in sorted(by_width.items()):
    n = d["null_id"] + d["raised"] + d["answered"]
    print(
        f"  width {band}: rows {n}, null_id {d['null_id']}, raised {d['raised']},"
        f" answered {d['answered']}, answered with -0.0 or a finite extreme {d['extreme']}"
    )
