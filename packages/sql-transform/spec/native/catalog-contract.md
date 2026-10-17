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

*Evidence:* `native/catalog_test.py::test_an_entry_matches_its_twin`, `native/catalog_test.py::test_an_output_field_of_one_arm_reads_no_id`.

**claim: entry-same-answer.** **The same answer.** By default, an entry
serves only where it is bit-exact. It is bit-exact where it does the twin's
operations in the twin's order, or where a kernel probe reads 0.

- An entry with a parity bound above 0 serves only when the caller asks
  for it, with `to_native(step, allow_bound=True)`. Such an entry can
  change a prediction: HistGradientBoosting flips labels on repeated
  training values.
- A parity bound is K·eps·S + τ for each output field. S is the error
  scale that the family declares, and K is derived, not measured.
- A family can compare a map of the two answers, and not the answers
  themselves. A distance to a fitted centre compares the two squares,
  because near the centre the distance itself has no small bound. A step
  that compares through a map is never part of a composition:
  `to_native` returns such a composition unchanged.
- In a composition, each output field has the parity bound of the step or
  part that gives it. With `allow_bound=True`, a step with a parity bound
  above 0 can be a step of a `Pipeline`, or a part of a
  `ColumnTransformer` or a `FeatureUnion`. Where the step reads the same
  input as the step inside the twin, its own bound holds. Each later step
  that reads its output gives its own output fields a carried bound. The
  affine steps give one: the scalers except `Normalizer`, the identity and
  the exact functions of `FunctionTransformer`, the feature selectors, the
  mean of `FeatureAgglomeration` and the linear projections. A weight in
  `transformer_weights` gives a part a carried bound too. If another step
  reads the output of a step with a bound above 0, `to_native` returns the
  `PythonTransform` unchanged. A discretizer is an example of such a step.
  `to_native` does the same if an infinite weight multiplies such an
  output. The owner ruled this in
  [bounded-steps-in-compositions.md](../../../../loops/native/decisions/closed/bounded-steps-in-compositions.md).
- The owner ruled this in
  [native-transform-parity-bounds.md](../../../../loops/confit/decisions/closed/native-transform-parity-bounds.md)
  and [matvec-parity-bound.md](../../../../loops/native/decisions/closed/matvec-parity-bound.md).
- NaN equals NaN, and NULL equals NULL. With a bound of 0, -0.0 and 0.0
  are different answers.

*Evidence:* `native/catalog_test.py::test_an_entry_matches_its_twin`, `native/catalog_test.py::test_a_reordered_translation_is_caught`, `native/project_test.py::test_a_projection_parts_from_its_twin_within_its_error_scale`, `native/project_test.py::test_a_projection_serves_only_on_request`, `native/compose_test.py::test_a_bounded_step_serves_last_or_as_a_part_on_request`, `native/compose_test.py::test_a_bounded_step_is_carried_by_the_steps_after_it`, `native/compose_test.py::test_a_carried_bound_grows_as_the_step_that_carries_it`, `native/compose_test.py::test_a_weight_carries_its_part_s_bound_times_the_weight`, `native/compose_test.py::test_a_pipeline_names_the_step_it_refuses`, `native/check_test.py::test_a_composition_holds_its_last_step_to_its_error_scale`, `native/check_test.py::test_a_composition_carries_an_error_scale_a_later_step_reads`, `native/_carry_test.py::test_a_carry_bounds_the_two_sides`, `native/check_test.py::test_a_composition_holds_a_bit_exact_part_bit_exactly`, `native/distance_test.py::test_a_distance_near_a_centre_is_compared_by_its_square`, `native/distance_test.py::test_a_distance_in_a_composition_stays_python`.

**claim: entry-same-inputs.** **The same inputs.** A feature reaches the
translation as `PythonTransform` hands it to `transform`. A number or a
boolean is a DOUBLE, with NULL read as NaN. A string stays a string.

*Evidence:* `native/catalog_test.py::test_an_entry_matches_its_twin`, `native/catalog_test.py::test_an_encoder_reads_a_string_feature_missing_at_fit`.

**claim: entry-sparse-outputs.** **Sparse outputs.** The twin reads a
sparse output as its dense array
([claim: sparse-output-dense](../python/estimator-data.md)). The entry
gives that dense answer.

- Inside a `Pipeline`, a step with a sparse output hands the next step the
  sparse matrix. The next step then uses its own code for a sparse input,
  and that code can answer differently from its code for a dense input.
- So the entry serves such a `Pipeline` only if both of these conditions
  are true:
  1. Each step that has a sparse output, except the last step, hands on
     only 0s and 1s.
  2. Each step that is handed such a sparse matrix of 0s and 1s answers it
     as it answers the dense array of the same values.
- If a `Pipeline` does not meet these conditions, `to_native` returns the
  step unchanged ([claim: to-native-best-effort](catalog-contract.md)).
  `to_native(step, strict=True)` raises `NotNative`, which names the step
  of the `Pipeline` and the reason.
- The twin never reads -0.0 from a sparse output, because `.toarray()`
  adds each stored value to 0.0. But a step inside a `Pipeline`,
  `ColumnTransformer` or `FeatureUnion` can compute -0.0. So the entry
  adds 0.0 to each output field of a `Pipeline`, `ColumnTransformer` or
  `FeatureUnion` that has a sparse output. Adding 0.0 changes -0.0 into
  0.0 and keeps every other value.
- A `ColumnTransformer` or `FeatureUnion` can multiply the output of a
  part by a weight (`transformer_weights`). If that part has a sparse
  output, the entry also adds 0.0 to each output field of the part, after
  the weight.

*Evidence:* `native/sparse_test.py::test_a_sparse_output_is_served_on_every_row`, `native/sparse_test.py::test_a_sparse_output_that_stays_python_says_why`.

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
- In a composition, a step can read the output of a step with a parity
  bound above 0. On a row of values near the largest double, a sum of the
  bounded step can overflow on one side only. Then the validation of the
  later step raises an error on that side only. The twin can raise where
  the entry answers, or answer where the entry traps. On such rows this
  claim does not hold, and `native.check` reports a `ParityError`. The twin
  itself raises on such a row in one call and answers in another. An open
  decision record asks the owner what must hold there:
  [traps-where-a-sum-overflows.md](../../../../loops/native/decisions/open/traps-where-a-sum-overflows.md).

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
