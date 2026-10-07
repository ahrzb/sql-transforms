# Native catalog contract

`to_native(step)` turns a fitted `PythonTransform` into a confit
`SqlFunction`. The function serves the same calls, with no Python in the
work for each row. The loop that grows the catalog is in
[loops/native/](../../../../loops/native/README.md).

## What an entry must equal

A catalog entry replaces its twin, the `PythonTransform` that the entry was
made from. The entry gives:

**claim: entry-same-call.** **The same call.** It has the same name. It
takes the instance id first, then the declared features, by position. It
returns the same type: a scalar, a struct with the same field names, or a
list of the same width.

*Evidence:* `native/catalog_test.py::test_an_entry_matches_its_twin`, `native/catalog_test.py::test_a_lane_of_one_arm_reads_no_id`.

**claim: entry-same-answer.** **The same answer.** By default, an entry
serves only where it is bit-exact. It is bit-exact where it does the twin's
operations in the twin's order, or where a kernel probe reads 0.

- An entry with a parity bound above 0 serves only when the caller asks
  for it, with `to_native(step, allow_bound=True)`. Such an entry can
  change a prediction: HistGradientBoosting flips labels on repeated
  training values.
- A parity bound is K·eps·S + τ for each output field. S is the error
  scale that the family declares, and K is derived, not measured.
- The owner ruled this in
  [native-transform-parity-bounds.md](../../../../loops/confit/decisions/closed/native-transform-parity-bounds.md)
  and [matvec-parity-bound.md](../../../../loops/native/decisions/closed/matvec-parity-bound.md).
- NaN equals NaN, and NULL equals NULL. With a bound of 0, -0.0 and 0.0
  are different answers.

*Evidence:* `native/catalog_test.py::test_an_entry_matches_its_twin`, `native/catalog_test.py::test_a_reordered_translation_is_caught`.

**claim: entry-same-inputs.** **The same inputs.** A feature reaches the
translation as `PythonTransform` hands it to `transform`. A number or a
boolean is a DOUBLE, with NULL read as NaN. A string stays a string.

*Evidence:* `native/catalog_test.py::test_an_entry_matches_its_twin`, `native/catalog_test.py::test_an_encoder_reads_a_string_feature_missing_at_fit`.

**claim: entry-same-instance-ids.** **The same instance ids.** If the
instance id is NULL, the entry answers NULL. For a struct return, this is a
NULL struct. An id that the step does not know raises an error.

*Evidence:* `native/catalog_test.py::test_an_unknown_id_raises_as_the_twin_does`, `native/catalog_test.py::test_a_null_id_is_a_null_struct`.

**claim: entry-swap-check.** **A check that swaps the entry.**
`native.check` serves one query with confit twice: once with the twin and
once with the entry. It also checks that the entry answers exactly what
DuckDB answers for the entry's own definition.

*Evidence:* `native/catalog_test.py::test_an_entry_matches_its_twin`, `native/catalog_test.py::test_a_reordered_translation_is_caught`, `native/guard_test.py::test_check_asserts_a_trap_where_the_step_raises_and_nowhere_else`.

## Converting a step

**claim: to-native-best-effort.** `to_native` does what it can. A step
with no translation comes back unchanged, and that step is always correct
to serve. `to_native(step, strict=True)` raises an error instead.

*Evidence:* `native/catalog_test.py::test_a_step_without_a_translation_comes_back_unchanged`, `native/catalog_test.py::test_to_native_is_idempotent`.

**claim: explicit-native-selection.** Native selection is explicit and has
no automatic cache. The native contract and parity bounds apply
independently of the authoring model.

*Evidence:* `native/check_test.py::test_an_error_scale_serves_only_on_request`, `native/catalog_test.py::test_a_step_without_a_translation_comes_back_unchanged`.

## Where the twin raises

**claim: entry-traps-where-twin-raises.** If the validation of the twin
raises on an input, the entry must trap on that input too. An input guard
in the entry does this. The owner ruled this in
[tolerated-differences.md](../../../../loops/native/decisions/closed/tolerated-differences.md).

- Until an entry has its input guard, it may answer where the twin raises.
- An entry may never trap where the twin answers.
- A twin error that is not validation, for example an sklearn bug, is not
  covered. The ruling record lists each such error.

*Evidence:* `native/guard_test.py::test_the_guard_traps_where_the_validation_raises`, `native/guard_test.py::test_check_asserts_a_trap_where_the_step_raises_and_nowhere_else`.

## Scope

**claim: catalog-scope.** A transformer is in scope if its `transform`
maps one row of features to one row of outputs, and uses only the fitted
state. These transformers are out of scope, and [coverage.md](coverage.md)
gives the reason for each:

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

*Evidence:* `native/coverage_test.py::test_every_named_class_has_a_row`, `native/coverage_test.py::test_the_coverage_table_is_current`.
