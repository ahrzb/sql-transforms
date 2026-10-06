"""How many of the gate's own rows (catalog_test._rows, 40 per step) `check`
leaves uncompared because the twin raises on them, per class, seeds 0..N-1."""
import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import shim  # noqa: F401
import collections, sys, time, warnings
warnings.simplefilter("ignore")
from threadpoolctl import threadpool_limits
from sql_transform.native import NotNative, check, to_native
from sql_transform.native.catalog_test import FIXTURES, _rows, _step

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
tot = collections.Counter(); cmp_ = collections.Counter(); steps = collections.Counter()
t0 = time.time()
with threadpool_limits(limits=1):
    for cls, fs in FIXTURES.items():
        for j, make in enumerate(fs):
            for seed in range(N):
                step = _step(make, seed, j)
                try:
                    native = to_native(step, strict=True)
                except NotNative:
                    continue
                rows = _rows(step, seed, getattr(make, "positive", False))
                n = check(step, native, rows)
                tot[cls.__name__] += rows.num_rows; cmp_[cls.__name__] += n; steps[cls.__name__] += 1
print(f"seeds 0..{N-1}, {time.time()-t0:.0f}s")
print("| class | steps | rows | uncompared (twin raised) | share |")
print("|---|---|---|---|---|")
for k in tot:
    u = tot[k] - cmp_[k]
    print(f"| {k} | {steps[k]} | {tot[k]} | {u} | {u/tot[k]:.1%} |")
T, C = sum(tot.values()), sum(cmp_.values())
print(f"| all | {sum(steps.values())} | {T} | {T-C} | {(T-C)/T:.1%} |")
