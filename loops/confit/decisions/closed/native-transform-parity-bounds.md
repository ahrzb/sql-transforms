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

**Ruling (owner, 2026-10-05).** Bit-equal is the target, relaxed slightly
where it cannot hold: floats do not align exactly when the operation order
differs. So a native entry is bit-exact wherever its operation order is the
twin's (the scaler and tree tiers, and any SQL-defined transform against
DuckDB), and within a small per-family ulp bound otherwise (matvec tiers,
whose sums sklearn orders through NumPy/BLAS). Each family declares its bound
with the measurement that fixed it, gated by swap-the-entry.

**Amended (owner, 2026-10-06).** A small ulp bound cannot hold where the
order of operations differs. The twin itself moves by 9,877,709,850 ulps
between a call with one row and a call with a batch. So a family whose
order cannot be the twin's declares a parity bound instead:

- For each output field, the entry is within K·eps·S + τ of the twin.
- S is the error scale of the field, a formula over the fitted state and
  the input row.
- K is derived from the error bounds of each operation. A measurement is
  evidence, not K.
- A ulp bound stays valid. It is a parity bound whose error scale is the
  size of the result.

The rulings that apply this amendment are in
`loops/native/decisions/closed/`: the matvec, power and additive chi2
records. Each of them gives its S and K.
