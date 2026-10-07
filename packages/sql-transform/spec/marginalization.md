# Window marginalization

## The accepted query

**claim: marginalize-entry-point.** Call `SQLProjection.marginalize(sql, connection=None, captured=None)` for the kept window convenience.
The ordinary constructor expects explicit fit/request SQL and does not marginalize automatically.

*Evidence:* `_marginal_test.py::test_the_law_holds_bit_for_bit_against_the_original_query`, `_marginal_test.py::test_divergence_is_only_at_misses`.

**claim: one-select-over-this.** The convenience accepts one SELECT over `__THIS__`, not a common-table-expression or derived-table projection chain.
It refuses source joins, WHERE, GROUP BY, HAVING, QUALIFY, SAMPLE, and set operations.
Top-level DISTINCT, ORDER BY, and LIMIT also refuse.
Use general SQLTransform for computations that change request rows.

*Evidence:* `_marginal_test.py::test_refusals_fire_pre_rewrite_in_the_authors_vocabulary`.

## Lookup keys

**claim: row-visible-window-keys.** Admitted window values must depend on row-visible keys, not physical row position.
Partition keys determine whole-partition values.
For moving RANGE or GROUPS frames, order values also become lookup keys.
An order key absent from fit data returns a lookup miss, not a newly evaluated window over requests.
COLLATE remains in the window expression but is removed from lookup keys.

*Evidence:* `_marginal_test.py::test_an_unseen_order_value_is_a_miss`, `_marginal_test.py::test_the_law_holds_for_the_window_vocabulary`.

## Admitted windows

**claim: window-admission-table.**

| Window feature | Admission boundary |
| --- | --- |
| SQL aggregates | FILTER, DISTINCT, ordered arguments, and serialized named windows remain supported |
| Frames | Whole-partition frames and admitted RANGE/GROUPS frames with valid constant bounds |
| Rank family | `rank`, `dense_rank`, `percent_rank`, and `cume_dist` |
| Value windows | Admitted `first_value`, `last_value`, and `nth_value`; nth offset must be a positive integer constant |
| IGNORE NULLS | Admitted value windows only, not general aggregates or raw fit windows |
| Physical-position functions | `row_number`, `ntile`, `lag`, and `lead` refuse |
| Other frame features | Bounded ROWS, EXCLUDE, invalid bounds, and unsupported offsets/defaults refuse |

*Evidence:* `_marginal_test.py::test_the_law_holds_for_the_window_vocabulary`, `_marginal_test.py::test_refusals_fire_pre_rewrite_in_the_authors_vocabulary`.

**claim: window-argument-limits.** Window fit arguments cannot contain nested windows, aggregates, subqueries, estimator bundles, or projection calls.
Positional column references and whole-row fit arguments also refuse.
Use explicit stages and named scalar features instead.

*Evidence:* `_marginal_test.py::test_refusals_fire_pre_rewrite_in_the_authors_vocabulary`.

## Subqueries

**claim: marginal-subqueries.** The convenience preserves uncorrelated scalar and EXISTS subqueries over fit data.
It rebinds their `__THIS__` reads to `__FIT__` and retains inner clauses.
Correlated subqueries, subqueries over other tables, and IN subqueries refuse in this convenience.
Use explicit authored SQL for supported core correlations.

*Evidence:* `_marginal_test.py::test_the_law_holds_for_uncorrelated_subqueries`, `_marginal_test.py::test_an_exists_answer_is_frozen_at_fit`, `_marginal_test.py::test_refusals_fire_pre_rewrite_in_the_authors_vocabulary`.

## Estimators in windows

**claim: estimator-window-forms.** A bare captured estimator call, `x(bundle)`, means global fit plus row-local application.
A partitioned estimator uses inline `x_transform(x_fit(bundle) OVER (...), bundle)`.

*Evidence:* `_raw_test.py::test_scaler_canonical_cte_and_window_have_the_same_public_artifact`, `_marginal_test.py::test_the_law_holds_for_projection_scopes`.

Standalone raw fit-window output refuses ([claim: refuses-standalone-fit-windows](unsupported-forms.md)).

**claim: raw-fit-window-limits.** Raw fit windows require whole partitions without window-clause running order.
Their FILTER and in-call ORDER BY remain supported.
Pure SQL projection fit windows exclude FILTER, DISTINCT, argument ORDER BY, IGNORE NULLS, and running order.
Use explicit SQL when a projection member needs those computations.

*Evidence:* `_raw_test.py::test_parked_raw_fit_window_names_the_inline_and_explicit_forms`, `_raw_test.py::test_ordered_fits_keep_input_order_for_ties_through_nested_wrappers`.

## Stars and aliases

**claim: stars-and-aliases.** Simple `*` and qualified request stars remain supported.
The derivation qualifies request stars so params fields cannot enter request output.
It preserves DuckDB's unaliased output names.
Ordinary scalar lateral aliases remain legal when they require no expansion into a fit expression.

*Evidence:* `_marginal_test.py::test_a_star_reads_the_batch_and_never_the_derived_params`, `_marginal_test.py::test_an_ordinary_lateral_alias_needs_no_fit_rewrite`.

A fit expression cannot read a sibling output alias ([claim: refuses-alias-expansion-into-fit](unsupported-forms.md)).

## Numerical fidelity

**claim: full-window-carrier.** The convenience builds one fit-time result table from the original executable SELECT items.
It retains window expressions, surrounding arithmetic, and their order.
It omits top-level star passthrough only from that private table, not from request output.
It appends ordinary window values and lookup keys, then selects distinct key/value columns for each scope.

This preserves the original window computation instead of replacing it with a GROUP BY aggregate.
DuckDB can change floating reduction order when windows execute separately or as grouped aggregates.
The numerical reference is the original admitted window SELECT over the fit data at `threads=1`.
Exact schemas and values include signed zero and NaN behavior.
Grouped SQL written by an author is lawful, but is not a universal bit-exact replacement for a window.

*Evidence:* `_marginal_test.py::test_the_carrier_keeps_the_original_reduction_order`, `_marginal_test.py::test_params_scale_with_fitted_keys_and_the_carrier_never_ships`.

**claim: raw-fits-independent-of-carrier.** Raw estimator fits remain independent of this private SQL-window table.
Mixed SQL-window and estimator queries keep the original ordinary windows, including windows inside raw feature expressions.
Raw feature, FILTER, and order columns do not become reads of the SQL-window table.

*Evidence:* `_raw_test.py::test_scaler_canonical_cte_and_window_have_the_same_public_artifact`, `_raw_test.py::test_inline_scope_sharing_does_not_merge_separately_authored_fits`.

## Lookups and NULL keys

**claim: window-lookups.** Window lookups match keys with `IS NOT DISTINCT FROM`.
NULL partition keys form one fitted partition.
A missing LEFT params match preserves the request and returns NULL values.
An empty SQL-window LEFT params table also returns NULL values.

*Evidence:* `_marginal_test.py::test_divergence_is_only_at_misses`, `_marginal_test.py::test_an_empty_fit_misses_every_request`, `_marginal_test.py::test_an_unseen_order_value_is_a_miss`.

Params join cardinality is in [claim: params-join-cardinality](projections.md).

```pycon
>>> import pyarrow as pa
>>> from sql_transform import SQLProjection
>>> fit_data = pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0]})
>>> requests = pa.table({"g": ["NEW", None, "a"], "v": [2.0, 14.0, 12.0]})
>>> fitted = SQLProjection.marginalize("SELECT g, v - avg(v) OVER (PARTITION BY g) AS d FROM __THIS__").fit(fit_data)
>>> expected = [{"g": "NEW", "d": None}, {"g": None, "d": 7.0}, {"g": "a", "d": -3.0}]
>>> assert fitted.transform(requests).to_pylist() == expected
>>> fn = fitted.compile()
>>> assert fn.infer_rows(requests.to_pylist()) == expected
>>> assert fn.infer_arrow(requests).to_pylist() == expected

```
