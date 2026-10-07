# The transform model

## Fit data and request data

**claim: fit-and-request-arguments.** A transform computes `(F, T) -> R` over relations.
`F` is the fit data, `T` is the request data, and `R` is the result.
Authored SQL names these arguments as `__FIT__` and `__THIS__`.
The author states the fit relationship in the SQL.

*Evidence:* `_freezing_test.py::test_pair_equals_composite`.

## Fit

**claim: fit-binds-fit-data.** Fit is partial application: it binds `F` and writes params tables.
The fitted transform then computes `T -> R` without reading the original fit relation.

*Evidence:* `_freezing_test.py::test_fit_ignores_this`.

[Window marginalization](marginalization.md) is a bounded convenience for writing this SQL.
It is not the definition of fit or the general model.

## General transforms and projections

**claim: general-transforms-are-compositional.** `SQLTransform` is the general, compositional model.
Its SQL can change the number of rows and can depend on several request rows.

*Evidence:* `_calling_test.py::test_chaining_fits_on_transformed_data`.

**claim: only-projections-compile.** `SQLProjection` adds a row-local rule: each request produces one answer independently of other requests.
Only `SQLProjection` offers Confit compilation.
([claim: row-local-rule](projections.md))

*Evidence:* `_projection_test.py::test_compile_returns_confits_own_function`, `_projection_test.py::test_a_foreign_leaf_serves_in_batch_but_refuses_to_compile_by_name`.

## Three separate checks

**claim: three-separate-checks.** These are separate checks:

1. The authoring model resolves and plans the SQL.
2. `SQLProjection` checks row-local SQL and the number of matches from params joins.
3. Confit either builds the fitted artifact or names a construct it does not serve.

A successful batch transform does not imply Confit admission.
Authoring admission does not extend the native catalog or Confit's SQL support.

*Evidence:* `_projection_test.py::test_refused_at_construction`, `_projection_test.py::test_a_multi_row_relation_beside_this_refuses_at_fit`, `_projection_test.py::test_a_foreign_leaf_serves_in_batch_but_refuses_to_compile_by_name`.

Relation callbacks stay batch-only ([claim: callbacks-batch-only](python/relation-callbacks.md)).

**claim: no-substitute-executor.** Compilation never substitutes another executor for a Confit refusal.

*Evidence:* `_projection_test.py::test_a_foreign_leaf_serves_in_batch_but_refuses_to_compile_by_name`.
