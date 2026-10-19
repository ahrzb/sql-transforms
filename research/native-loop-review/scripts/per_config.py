"""Per fixture configuration, over the gate's seeds 0..7: rows with an id
that the twin answers (the rows that test the translation), and the share of
NULL-id rows. Prints the configurations with the fewest."""
import inspect
import warnings

import sql_transform.native.catalog_test as ct

warnings.simplefilter("ignore")
out = []
for cls, makes in ct.FIXTURES.items():
    for j, make in enumerate(makes):
        real = nulls = total = steps = 0
        for seed in range(8):
            try:
                step = ct._step(make, seed, j)
            except Exception:
                continue
            steps += 1
            for row in ct._rows(step, seed, getattr(make, "positive", False)).to_pylist():
                total += 1
                iid = row.pop("__iid")
                if iid is None:
                    nulls += 1
                    continue
                try:
                    step(iid, *row.values())
                    real += 1
                except Exception:
                    pass
        try:
            src = inspect.getsource(make).strip().splitlines()[0][:90]
        except (TypeError, OSError):
            src = repr(make)[:90]
        out.append((real, cls.__name__, j, steps, total, nulls, src))
out.sort()
print("configurations:", len(out))
print("with <= 20 answered rows with an id over 8 seeds:", sum(1 for o in out if o[0] <= 20))
for real, name, j, steps, total, nulls, src in out[:25]:
    print(f"{name}-{j}: answered {real}/{total} rows (NULL id {nulls}), steps {steps}: {src}")
