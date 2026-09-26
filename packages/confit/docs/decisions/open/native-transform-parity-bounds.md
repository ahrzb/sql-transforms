# Parity bounds for native transform families

**Question.** A native transform entry (typed `StandardScaler`, `PCA`, trees and
so on, behind the existing extern slots) replaces a call to sklearn's
`transform()`. What must it equal?

**Proposed bound.** A native entry equals its `PythonTransform` twin: bit-exact
for scaler and tree tiers, within a declared per-family ulp bound for matvec
tiers, gated by swap-the-entry (same SQL, same statics, a different entry in the
UDF list).

**Why it matters.** A fitted transformer costs about 118 µs per row against 1.4 µs
for the same query without it (`benchmarks/bench_transforms.py`, 2026-09-26), and
nearly all of it is sklearn's `transform()`. Native families wait on this bound:
kpi: transformer-parity (C4) moves only by reviewed decision.

**Ruling.** None yet.
