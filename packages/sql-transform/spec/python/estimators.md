# Estimators and scalar UDFs

## What a projection may capture

**claim: projection-only-python.** Only `SQLProjection` admits captured raw row-local estimators and declared scalar user-defined functions (UDFs).
General `SQLTransform` admits explicit relation callbacks instead.
An estimator capture must be an instance with callable `fit` and `transform` methods, not an estimator class.
Here, “raw estimator” means that captured instance, rather than an explicit `Transform` callback.

*Evidence:* `_resolution_test.py::test_projection_only_python_admission_does_not_change_general_run`.

**claim: scalar-udf-declarations.** `UDF` declares a scalar serving function's input and output types.
`PythonUDF` wraps a Python scalar callable with those declarations.
Declared names, arity, input types, and return types remain authoritative.
Case-insensitive builtin collisions and conflicting fit/transform names refuse.

*Evidence:* `_raw_test.py::test_scalar_udfs_keep_exact_large_integers_and_nulls`, `_resolution_test.py::test_scalar_arity_and_sql_name_collisions_refuse_at_compile`.

## The params query

**claim: canonical-raw-params-query.** A raw params query selects GROUP BY keys and exactly one `x_fit(bundle)` item.
Its source is direct `__FIT__` or one private inline SELECT over direct `__FIT__`.
The params query can have GROUP BY, but no other selected computations or query clauses.

The inline source can project ordinary windows and estimator-free scalar subqueries.
It cannot add joins, WHERE, GROUP BY, collapsing aggregates, or another source level.
Both levels exclude HAVING, QUALIFY, DISTINCT, set operations, ORDER BY, and LIMIT.
These source limits do not remove clauses inside an admitted estimator-free scalar subquery.

*Evidence:* `_resolution_test.py::test_noncanonical_raw_params_refuse_at_compile`, `_resolution_test.py::test_raw_replay_retains_only_author_captures_and_fits_one_scope_once`.

## Applying a fitted estimator

**claim: raw-application-forms.** The params query must be a common table expression or derived table read directly by the request SELECT.
It cannot hide beneath another fit-only query.
A request application consumes that source's direct ID column in the same lexical scope.
The other legal application form is an inline canonical scalar fit subquery.
Arbitrary ID expressions and intermediate ID passthrough do not select a learned schema.

*Evidence:* `_resolution_test.py::test_inline_raw_fit_is_bound_without_changing_public_run`, `_resolution_test.py::test_raw_ids_cannot_be_arbitrary_expressions`.

## Bundles

**claim: bundle-forms.** A bundle is a named scalar `struct_pack(...)`, or a bare column named by its final path component.
Fit normalizes a bare column to a named struct.
Application fields must match the fit fields in order.
Nested estimator bundles, whole rows, nested feature structures, and feature field access refuse.
Select learned output fields explicitly, or keep the ordinary output struct.

*Evidence:* `_resolution_test.py::test_raw_bundle_fields_are_ordered_and_named`, `_raw_test.py::test_mixed_arrow_schema_controls_fit_and_serving_conversion`.

## Fit scopes

**claim: independent-fit-scopes.** Each separately authored explicit fit source fits independently, even when its SQL text repeats.
One fit source can serve several applications.
Window marginalization alone coalesces equivalent inline fits by their printed source, keys, bundle fields, FILTER, and order.
Raw params stay separate from ordinary SQL-window params.

*Evidence:* `_resolution_test.py::test_separately_authored_raw_sources_have_separate_bindings_and_instances`, `_raw_test.py::test_inline_scope_sharing_does_not_merge_separately_authored_fits`.

## Learned UDFs and IDs

**claim: learned-udfs-per-scope.** Fit creates per-scope IDs and publishes `PythonTransform` UDF objects.
Each such object wraps its scope's fitted estimator instances.
Fit publishes each learned UDF before a later fit step or the serving SQL binds.
Each UDF contains only its scope's ID-to-estimator mapping.
The artifact's `instances` mapping references those same estimator objects.

*Evidence:* `_resolution_test.py::test_raw_replay_retains_only_author_captures_and_fits_one_scope_once`, `_resolution_test.py::test_separately_authored_raw_sources_have_separate_bindings_and_instances`.

**claim: ids-select-schemas.** IDs are schema selectors, not authentication.
Unknown runtime IDs retain their named error behavior.
The positive canonical ID grammar does not trace forged structs, altered IDs, or cross-scope expressions.
No exporter, runtime authentication, or intermediate-handle tracing extends that grammar.

*Evidence:* `_resolution_test.py::test_raw_ids_cannot_be_arbitrary_expressions`, `_raw_test.py::test_scaler_canonical_cte_and_window_have_the_same_public_artifact`.

**claim: display-structs-are-not-handles.** A display struct such as `struct_pack(type := 'sc', id := p.iid)` is ordinary SQL, not a consumable raw handle.
The examples name a params ID column `iid`.
That column name is an authored alias, not a special handle.

*Evidence:* `_resolution_test.py::test_raw_ids_cannot_be_arbitrary_expressions`.

## Raw fit refusals

**claim: raw-fit-refusals.** Raw fit also refuses request-data refitting, applications inside fit-only subtrees, nested estimators, and learned-output-to-later-fit composition.

*Evidence:* `_resolution_test.py::test_noncanonical_raw_params_refuse_at_compile`, `_resolution_test.py::test_raw_scope_passthrough_windows_and_fit_only_apply_refuse`.

Other raw fit sources refuse too ([claim: refuses-noncanonical-raw-fit-sources](../unsupported-forms.md)).

## Examples

### One scaler per key

This example fits one scaler per key.
An unseen key returns NULL, and the NULL key matches its fitted group.
The four public serving fields reconstruct Confit without private state.
Strict native conversion must produce native entries or raise `NotNative`.
It cannot silently retain Python execution.

```pycon
>>> import pyarrow as pa
>>> from sklearn.preprocessing import StandardScaler
>>> from confit import DuckDBInferFn
>>> from sql_transform import PythonTransform, SQLProjection
>>> from sql_transform.native import to_native
>>> fit_data = pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0]})
>>> requests = pa.table({"g": ["a", "NEW", None], "v": [20.0, 4.0, 14.0]})
>>> projection = SQLProjection('''
... WITH p AS (SELECT g, sc_fit(v) AS iid FROM __FIT__ GROUP BY g)
... SELECT t.g, sc_transform(p.iid, t.v).v AS z
... FROM __THIS__ t LEFT JOIN p ON t.g IS NOT DISTINCT FROM p.g
... ''', captured={"sc": StandardScaler()})
>>> fitted = projection.fit(fit_data)
>>> expected = [{"g": "a", "z": 1.0}, {"g": "NEW", "z": None}, {"g": None, "z": 7.0}]
>>> assert fitted.transform(requests).to_pylist() == expected
>>> assert all(isinstance(udf, PythonTransform) for udf in fitted.udfs.values())
>>> assert all(fitted.instances[iid] is instance for udf in fitted.udfs.values() for iid, instance in udf.instances.items())
>>> public = DuckDBInferFn(sql=fitted.sql, row_tables={"__THIS__": fitted.schema}, static_tables=fitted.params, udfs=list(fitted.udfs.values()), shape="map")
>>> assert public.infer_rows(requests.to_pylist()) == expected
>>> assert public.infer_arrow(requests).to_pylist() == expected
>>> native_udfs = [to_native(udf, strict=True) for udf in fitted.udfs.values()]
>>> native = DuckDBInferFn(sql=fitted.sql, row_tables={"__THIS__": fitted.schema}, static_tables=fitted.params, udfs=native_udfs, shape="map")
>>> assert native.infer_rows(requests.to_pylist()) == expected
>>> assert native.infer_arrow(requests).to_pylist() == expected
>>> replay = SQLProjection(projection.source, captured=projection.captured).fit(fit_data)
>>> assert replay.transform(requests).to_pylist() == expected

```

Strict conversion follows the [native catalog contract](../native/catalog-contract.md).

### A window-derived fit feature

A raw inline source can compute a window-derived feature over the original fit data.
The request expression must use the same fitted mean.
Here the mean comes from a DISTINCT pick of the original window values, not a replacement GROUP BY reduction.

```pycon
>>> import pyarrow as pa
>>> from sklearn.preprocessing import StandardScaler
>>> from sql_transform import SQLProjection
>>> fit_data = pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0]})
>>> requests = pa.table({"g": ["a", "NEW", None], "v": [20.0, 4.0, 14.0]})
>>> fitted = SQLProjection('''
... WITH c AS (SELECT g, avg(v) OVER (PARTITION BY g) AS m FROM __FIT__),
... means AS (SELECT DISTINCT c.g, c.m FROM c),
... p AS (SELECT g, sc_fit(cv) AS iid
...       FROM (SELECT g, v - avg(v) OVER (PARTITION BY g) AS cv FROM __FIT__) f
...       GROUP BY g)
... SELECT t.g, sc_transform(p.iid, struct_pack(cv := t.v - means.m)).cv AS z
... FROM __THIS__ t
... LEFT JOIN means ON t.g IS NOT DISTINCT FROM means.g
... LEFT JOIN p ON t.g IS NOT DISTINCT FROM p.g
... ''', captured={"sc": StandardScaler()}).fit(fit_data)
>>> expected = [{"g": "a", "z": 1.0}, {"g": "NEW", "z": None}, {"g": None, "z": 7.0}]
>>> assert fitted.transform(requests).to_pylist() == expected
>>> assert fitted.compile().infer_rows(requests.to_pylist()) == expected
>>> assert fitted.compile().infer_arrow(requests).to_pylist() == expected

```
