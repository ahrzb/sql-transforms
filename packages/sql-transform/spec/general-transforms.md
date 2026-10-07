# General transforms

## Constructor

**claim: sqltransform-constructor.** Use `SQLTransform(sql, output='default', connection=None, captured=None)` for general SQL.

*Evidence:* `_catalog_test.py::test_every_declared_output_is_accepted`.

## Fit and the fitted artifact

**claim: fit-returns-fitted.** `fit(data)` returns `Fitted`, not the estimator itself.
The estimator also remembers that artifact for later `transform(data)` calls.

*Evidence:* `_freezing_test.py::test_fitted_is_callable_and_transform_is_the_same_thing`.

**claim: sklearn-interface.** `fit_transform`, `get_params`, `set_params`, sklearn cloning, and `set_output` use the same program.
The optional sklearn `y` argument is ignored.
Put targets in columns that the authored SQL can read.

*Evidence:* `_boundary_test.py::test_clone_survives_a_shared_connection`, `_boundary_test.py::test_cross_val_score_runs_with_a_shared_connection`.

## Output modes

**claim: output-modes.**

| Estimator output mode | Result of `SQLTransform.transform` |
| --- | --- |
| `default` or `arrow` | Materialized `pyarrow.Table` |
| `duckdb` | Lazy DuckDB relation |
| `pandas` | pandas DataFrame |
| `numpy` | NumPy array |

`Fitted.transform` always returns Arrow.

*Evidence:* `_catalog_test.py::test_as_output_handles_every_declared_output`, `_connection_test.py::test_duckdb_output_is_a_relation_that_agrees_with_arrow`, `_boundary_test.py::test_pandas_output_carries_the_callers_index`.

**claim: fitted-relation.** `Fitted.relation` returns a lazy relation, and `Fitted` is callable as `transform`.

*Evidence:* `_connection_test.py::test_duckdb_output_is_a_relation_that_agrees_with_arrow`.

**claim: pandas-index-alignment.** For pandas output, the estimator preserves the source index only when row counts agree and no top-level ordering or limit breaks alignment.

*Evidence:* `_boundary_test.py::test_pandas_output_carries_the_callers_index`.

General SQL has no implicit row-order promise.

## Lazy relations

**claim: lazy-relation-lifetime.** A lazy relation belongs to the connection that built it.
Use the same caller connection for lazy chaining.
Keep the fitted artifact alive while consuming its lazy relations, including derived relations.
The artifact retains their table and function leases until `release()` or finalization.
`release()` is idempotent and invalidates outstanding relations that still need those registrations.
Use eager transforms for repeated serving without retained lazy registrations.

*Evidence:* `_lifecycle_test.py::test_a_lazy_relation_survives_a_later_execution`, `_lifecycle_test.py::test_lazy_retention_is_bounded_by_the_artifact`, `_lease_test.py::test_release_gives_them_back_on_demand`.
