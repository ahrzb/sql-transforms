"""The parity test's steps (seeds 0..7) where the twin raises on every row
with an id: `check` still counts the NULL-id rows as compared. For each, does
`to_native` serve it (so `check` runs and passes on NULL rows alone)?"""
import warnings
from collections import Counter

import sql_transform.native.catalog_test as ct
from sql_transform.native import NotNative, to_native
from sql_transform.native._check import check

warnings.simplefilter("ignore")
hits = Counter()
for cls, makes in ct.FIXTURES.items():
    for j, make in enumerate(makes):
        for seed in range(8):
            try:
                step = ct._step(make, seed, j)
            except Exception:
                continue
            table = ct._rows(step, seed, getattr(make, "positive", False))
            real = 0
            for row in table.to_pylist():
                iid = row.pop("__iid")
                if iid is None:
                    continue
                try:
                    step(iid, *row.values())
                    real += 1
                except Exception:
                    pass
            if real:
                continue
            try:
                native = to_native(step, strict=True, allow_bound=True)
            except NotNative as e:
                print(f"{cls.__name__}-{j}-{seed}: skipped ({str(e)[:80]})")
                continue
            n = check(step, native, table)
            hits[cls.__name__] += 1
            print(f"{cls.__name__}-{j}-{seed}: width {len(step.takes)}, check passed, 'compared' {n} (all NULL-id)")
print(dict(hits))
