# The native catalog: goal

`to_native(step)` turns a fitted `PythonTransform` into a confit
`SqlFunction`. The function serves the same calls, with no Python in the
work for each row. Through sklearn, a fitted transformer costs about 118 µs
for each row. The same query without the transformer costs about 1.4 µs
(`benchmarks/bench_transforms.py`, 2026-09-26). The native catalog removes
that cost for each transformer that it covers.

## The contract

A catalog entry replaces its twin, the `PythonTransform` that the entry was
made from. The entry gives:

- **The same call.** It has the same name. It takes the instance id first,
  then the declared features, by position. It returns the same type: a
  scalar, a struct with the same field names, or a list of the same width.
- **The same answer.** The answer is bit-exact where the entry does the
  twin's operations in the twin's order. Otherwise it is within a declared
  ulp bound for each family. The owner ruled this in
  [native-transform-parity-bounds.md](../confit/decisions/closed/native-transform-parity-bounds.md).
  NaN equals NaN, and NULL equals NULL. With a bound of 0, -0.0 and 0.0
  are different answers.
- **The same inputs.** A feature reaches the translation as
  `PythonTransform` hands it to `transform`. A number or a boolean is a
  DOUBLE, with NULL read as NaN. A string stays a string.
- **The same instance ids.** If the instance id is NULL, the entry
  answers NULL. For a struct return, this is a NULL struct. An id that the
  step does not know raises an error.
- **A check that swaps the entry.** `native.check` serves one query with
  confit twice: once with the twin and once with the entry. It also checks
  that the entry answers exactly what DuckDB answers for the entry's own
  definition.

`to_native` does what it can. A step with no translation comes back
unchanged, and that step is always correct to serve.
`to_native(step, strict=True)` raises an error instead.

## Tolerated differences

This rule is provisional. Its decision record
([tolerated-differences.md](decisions/open/tolerated-differences.md)) names
the condition that ends the rule.

- **Where the twin raises an error, the entry may answer.** This applies to
  input that sklearn's validation rejects, such as an infinite value for
  most estimators. The reverse is not allowed: an entry may not raise where
  the twin answers.

## Scope

A transformer is in scope if its `transform` maps one row of features to
one row of outputs, and uses only the fitted state. These transformers are
out of scope, and [coverage.md](coverage.md) gives the reason for each:

- transformers whose input is not a row of columns, such as text, dicts,
  images and kernel matrices;
- transformers of the target, such as `LabelEncoder`, which encode labels
  and not features;
- transformers with no `transform` for new rows.

The catalog serves compositions by composing entries. A composition is not
an entry of its own. The compositions are `Pipeline`, `ColumnTransformer`,
`FeatureUnion`, stacking and voting.

A transformer whose `transform` reads the rows that it was fitted on is in
scope. Two examples are nearest neighbours and kernel approximations over
the fit set. Such a transformer may need a capability that confit does not
have yet.
