# Estimator data

## Feature types

**claim: feature-type-widening.** BIGINT and DOUBLE are DuckDB's 64-bit integer and floating types.
Arrow names the corresponding declarations `int64` and `float64`.

Raw fit reads the actual Arrow bundle schema.
Integer declarations widen to int64, floating declarations widen to float64, and boolean declarations remain boolean.
String and large-string declarations become string.
Other feature types refuse by name.
Generated application calls cast features to these learned declarations.

*Evidence:* `_raw_test.py::test_narrow_features_publish_normalized_declarations`, `_raw_test.py::test_sql_cast_fit_features_publish_normalized_declarations`, `_named_outputs_test.py::test_mixed_feature_types`.

## Values at the estimator boundary

**claim: estimator-value-conversion.** At the estimator boundary, numeric values become Python floats and numeric NULL becomes NaN.
Numeric-only matrices use float64.
`dtype` is a NumPy array's element-type declaration.
Matrices with string or boolean fields use object dtype and preserve those values.

*Evidence:* `_raw_test.py::test_mixed_arrow_schema_controls_fit_and_serving_conversion`, `_raw_test.py::test_filtered_numeric_looking_strings_remain_strings`.

**claim: numeric-strings-stay-strings.** Numeric-looking strings stay strings.
Use an explicit SQL CAST when numeric conversion is intended.

*Evidence:* `_raw_test.py::test_filtered_numeric_looking_strings_remain_strings`.

**claim: integer-precision-at-boundary.** Integer values above `2**53` can lose precision at the numeric estimator boundary.
A declared integer scalar UDF does not inherit this estimator conversion.

*Evidence:* `_raw_test.py::test_mixed_arrow_schema_controls_fit_and_serving_conversion`, `_raw_test.py::test_scalar_udfs_keep_exact_large_integers_and_nulls`.

## Learned outputs

**claim: learned-output-struct.** Learned outputs are a DOUBLE struct.
Output width is the number of fields in that struct.
Fit validates one-row output shape and requires equal output schemas across groups within one scope.
`Named` wraps an estimator with authoritative output labels, including width-one labels.
Estimator feature names supply labels when usable; otherwise labels are `f0`, `f1`, and so on.

*Evidence:* `_named_outputs_test.py::test_bare_call_serves_width1_struct`, `_struct_outputs_test.py::test_bare_call_serves_struct_column_batch`, `_struct_outputs_test.py::test_bare_call_serves_struct_column_row_path`.

**claim: learned-output-refusals.** Missing fields, case-colliding fields, and unsupported output shapes refuse.
Different scopes can learn different widths from the same prototype.

*Evidence:* `_named_outputs_test.py::test_refit_that_drops_a_field_refuses_by_name`, `_named_outputs_test.py::test_per_group_shape_disagreement_refuses`, `_named_outputs_test.py::test_wide_call_in_list_extract_refuses_when_bound`.

## Fit order

**claim: fit-order-and-ties.** FILTER and in-call ORDER BY belong on the fit item.
Fit preserves declared directions, NULL placement, and collation, then uses materialized input position to order ties.
Unordered raw fits also preserve materialized input position.

*Evidence:* `_raw_test.py::test_ordered_fits_keep_input_order_for_ties_through_nested_wrappers`, `_ordered_test.py::test_desc_and_nulls_follow_duckdb_defaults`, `_ordered_test.py::test_multi_key_order_and_stable_ties`.

**claim: order-sensitive-requires-order.** `OrderSensitive`, including nested `Named` wrappers, requires in-call ORDER BY.
`OrderSensitive` is the wrapper that declares this ordering requirement.

*Evidence:* `_resolution_test.py::test_order_sensitive_requires_argument_order_not_a_window_order`, `_ordered_test.py::test_named_wrapping_order_sensitive_still_requires_order`.

**claim: running-fit-refuses.** An ORDER BY in the fit window clause is a running fit and refuses.
Literal orders remain literal and follow normal DuckDB binding policy.
A DuckDB relation's fit order means only its materialized result order.

*Evidence:* `_resolution_test.py::test_raw_scope_passthrough_windows_and_fit_only_apply_refuse`, `_raw_test.py::test_parked_raw_fit_window_names_the_inline_and_explicit_forms`.

## Groups and empty fits

**claim: grouped-list-fit.** Raw fit materializes its source once and collects complete groups through Arrow list aggregates.
It uses omitted Arrow parameter declarations and returns BIGINT IDs.

*Evidence:* `_raw_test.py::test_filtered_empty_group_mints_no_instance_and_misses`, `_raw_test.py::test_mixed_arrow_schema_controls_fit_and_serving_conversion`.

**claim: empty-groups.** Filtered-empty groups create no estimator instance, and fit removes their NULL-ID rows without fitting again.
Empty fit data and entirely filtered scopes raise `TransformError`.
A missing params match supplies a NULL ID and produces a NULL result.

*Evidence:* `_raw_test.py::test_filtered_empty_group_mints_no_instance_and_misses`, `_filter_test.py::test_partitioned_filtered_fit_unseen_group_is_null`.

## Reserved columns

**claim: reserved-ordinal-columns.** Raw fit refuses `__cf_fit_row` and `__cf_row` fit columns, case-insensitively, before adding private ordinals.
Batch projections also reserve `__cf_row` in request data.
Ordinary single-underscore names remain ordinary columns.
Private ordinals never become public output columns.

*Evidence:* `_raw_test.py::test_fit_ordinal_collisions_refuse_before_numbering`, `_resolution_test.py::test_raw_fit_reserved_ordinals_refuse_before_numbering`.
