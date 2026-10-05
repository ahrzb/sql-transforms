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
- **Same answer.** Bit-exact wherever the entry performs the twin's
  operations in the twin's order, and within a declared per-family ulp
  bound otherwise (owner ruling:
  `packages/confit/docs/decisions/closed/native-transform-parity-bounds.md`).
  NaN equals NaN; NULL equals NULL; with bound 0, -0.0 is not 0.0.
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

## Tolerated differences

Provisional, with the condition that ends it; see [decisions/](decisions/).

- **Where the twin raises,** on input sklearn's validation rejects (an
  infinite value for most estimators), the entry may answer. Never the
  reverse: an entry may not raise where the twin answers.

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
