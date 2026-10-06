# Native transforms: goal

`to_native(step)` turns a fitted `PythonTransform` into a confit
`SqlFunction` that serves the same calls with no Python on the row path. A
fitted transformer costs about 118 µs per row through sklearn against about
1.4 µs for the same query without it (`benchmarks/bench_transforms.py`,
2026-09-26); the catalog removes that cost for every transformer it covers.

## The contract

A native entry is a drop-in replacement for its twin, the `PythonTransform`
it was made from:

- **Same call.** Same name, the instance id first, then the declared
  features, by position. The same return type: a scalar, a struct with
  the same field names, or a list of the same width.
- **Same answer.** By default, an entry serves only where it is bit-exact.
  It is bit-exact where it does the twin's operations in the twin's order,
  or where a kernel probe reads 0.
  - An entry with a parity bound above 0 serves only when the caller asks
    for it, with `to_native(step, allow_bound=True)`. Such an entry can
    change a prediction: HistGradientBoosting flips labels on repeated
    training values.
  - A parity bound is K·eps·S + τ for each output field. S is the error
    scale that the family declares, and K is derived, not measured.
  - The owner rulings are in
    `loops/confit/decisions/closed/native-transform-parity-bounds.md` and
    [decisions/closed/matvec-parity-bound.md](decisions/closed/matvec-parity-bound.md).
  - NaN equals NaN, and NULL equals NULL. With bound 0, -0.0 is not 0.0.
- **Same inputs.** A feature reaches the translation as `PythonTransform`
  hands it to `transform`: a number (or boolean) as a DOUBLE with NULL read
  as NaN, a string as is.
- **Same instance ids.** A NULL instance id answers NULL (a NULL struct,
  for a struct return); an id the step does not know raises.
- **Gated by swap-the-entry.** One query, served by confit with the twin and
  with the entry (`native.check`); and the entry answers exactly what DuckDB
  answers for its own definition.

`to_native` is best effort: a step with no translation comes back unchanged,
which is always correct to serve. `to_native(step, strict=True)` raises
instead.

## Where the twin raises

If the validation of the twin raises on an input, the entry must trap on
that input too. An input guard in the entry does this. The owner ruling is
in [decisions/closed/tolerated-differences.md](decisions/closed/tolerated-differences.md).

- Until an entry has its input guard, it may answer where the twin raises.
- An entry may never trap where the twin answers.
- A twin error that is not validation, for example an sklearn bug, is not
  covered. The ruling record lists each such error.

## Scope

In scope: every sklearn transformer whose `transform` maps one row of
features to one row of outputs using only fitted state. Out of scope, with
the reason in [coverage.md](coverage.md): transformers whose input is not a
row of columns (text, dicts, images, kernel matrices), target encoders, and
those with no `transform` for new rows. Compositions (`ColumnTransformer`,
`FeatureUnion`, stacking and voting) are served by composing entries, not
as entries of their own.

A transformer whose `transform` reads its fitted samples (nearest
neighbours, kernel approximations over the fit set) is in scope; it may need
a capability confit does not have yet.
