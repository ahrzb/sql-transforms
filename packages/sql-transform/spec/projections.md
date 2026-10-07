# Projections

## The row-local rule

**claim: row-local-rule.** A projection serves one output row for each
request row, computed from that row and the params tables alone.
`SQLProjection` checks the SQL that remains after freezing, at
construction. Each query level that carries the request rows must be a
projection over joins that keep rows. A level that reads only params tables
is free, because it is a constant table at serving. A projection that
breaks the rule raises `NotRowWise`. Its `reason` is one of these names:

| Reason | What breaks the rule |
|---|---|
| `aggregate` | An aggregate folds the request rows into one value. |
| `window` | A window reads the other request rows through its frame. |
| `group-by` | `GROUP BY` folds the request rows together. |
| `modifier` | `DISTINCT`, `ORDER BY` or `LIMIT` changes which rows come back, or how many. |
| `filter` | `WHERE` or `QUALIFY` drops rows, and a scalar UDF has no encoding for "no row here". |
| `this-twice` | `__THIS__` enters the row stream more than once, so rows multiply. |
| `set-operation` | A set operation stacks the request rows onto something else. |
| `recursive-cte` | A recursive CTE iterates over the request rows. |
| `join` | A join can drop or duplicate request rows. |
| `spine` | Another break: the SQL reads `__THIS__` in an expression instead of `FROM`, never reads `__THIS__`, uses a positional reference, or calls `unnest`. |

`SQLTransform` makes no such promise.

*Evidence:* `_projection_test.py::test_refused_at_construction`, `_projection_test.py::test_every_reason_is_exercised`, `_projection_test.py::test_rows_and_order_with_an_unseen_key`.

## Params join cardinality

**claim: params-join-cardinality.** A CROSS params side requires exactly one match.
Empty CROSS tables and duplicate-key matches refuse.
A request-preserving outer params side allows zero or one match.

*Evidence:* `_projection_test.py::test_a_join_without_group_by_refuses_at_fit_naming_the_key`, `_projection_test.py::test_a_multi_row_relation_beside_this_refuses_at_fit`, `_projection_test.py::test_an_empty_relation_beside_this_refuses_at_fit`.

Supported authored correlations keep their own empty-input and guard behavior ([claim: misses-and-guards-preserved](fit/decorrelation.md)).

## The fitted projection

**claim: fitted-projection-fields.** `SQLProjection.fit(data)` returns `FittedProjection`.
Its public fields have these meanings:

| Field or method | Meaning |
| --- | --- |
| `sql` | Standalone unordered serving SQL; no public batch ordinal |
| `schema` | Nullable request schema containing the fit columns the serving SQL reads |
| `params` | One stored mapping of normalized captured statics and serving-live learned tables |
| `udfs` | UDF objects under the names used by serving SQL |
| `instances` | Fitted estimator or callback instances keyed by ID |
| `transform(data)` | Arrow batch result in request order |
| `compile()` | A fresh Confit `DuckDBInferFn` with shape `map`, or an explicit refusal |

*Evidence:* `_projection_test.py::test_params_are_inspectable_and_fit_is_gone_from_the_text`, `_projection_test.py::test_public_artifact_reads_the_same_captured_snapshot_in_every_interface`.

**claim: public-serving-artifact.** For Confit-admitted projections, `sql`, `schema`, `params`, and `udfs` are the complete public serving artifact.
Read Confit metadata and call `infer_rows` or `infer_arrow` on the returned function.
The projection does not forward those interfaces.

*Evidence:* `_projection_test.py::test_compile_returns_confits_own_function`, `_projection_test.py::test_public_artifact_reads_the_same_captured_snapshot_in_every_interface`.

The [one scaler per key](python/estimators.md#one-scaler-per-key) example rebuilds a Confit function from the four public fields.

Direct execution of public serving SQL has no row-order promise.

**claim: batch-keeps-request-order.** Batch `transform` uses a private ordinal query to preserve request order.

*Evidence:* `_projection_test.py::test_the_input_order_is_the_output_order_not_the_join_order`, `_serving_test.py::test_transform_preserves_input_row_order_with_unseen_groups`.

## Request schema

**claim: request-schema-widths.** The request schema keeps real input widths.
Raw feature casts do not widen every request column.
Confit can therefore refuse a referenced float32 request column even when its learned estimator declaration is float64.
No declared-schema selection, missing-column policy, or extra-column dropping overrides normal DuckDB and Confit binding.

*Evidence:* not pinned by a test.

## Captured relations

**claim: captures-normalized-once.** Projection fit normalizes captured non-Arrow relations to Arrow once.
DuckDB relations use `to_arrow_table()`, not a reusable Arrow reader.
Fit, probes, batch, compilation, and public-field reconstruction all read the same stored `params` mapping.
Callers may replace mapping entries, but must treat underlying Arrow buffers as read-only.
Refit when a captured source intentionally changes.
General `Fitted.params` remains learned-only, with separate live captured bindings.

*Evidence:* `_projection_test.py::test_public_artifact_reads_the_same_captured_snapshot_in_every_interface`.

## Caller catalogs

**claim: catalog-tables-batch-only.** Caller-catalog tables remain batch-only unless the author captures an Arrow snapshot and refits.
Compilation refuses catalog dependencies because Confit has no caller catalog.
The package never silently captures catalog tables.

*Evidence:* `_projection_test.py::test_caller_catalog_probes_and_batch_restore_observed_threads`.

Execution uses the caller connection ([claim: caller-connection-execution](connections.md)).
