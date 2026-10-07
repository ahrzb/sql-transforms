# Relation callbacks

## The callback protocol

**claim: callback-invocation.** `Transform(fit, transform, takes, returns)` supplies two relation callbacks.
The fit callback receives a complete SQL aggregate group as an Arrow table.
It returns an instance that the fitted artifact retains.
The transform callback receives that instance and its rows within one Arrow invocation.
It does not receive the whole request relation or a fixed number of rows.

*Evidence:* `_foreign_test.py::test_the_pair_fits_and_serves`, `_connection_test.py::test_the_relation_is_not_executed_until_it_is_consumed`.

**claim: callback-declarations.** `takes` and `returns` declare ordered field names with authoritative DOUBLE types.
The callback must return those fields and one row per supplied row.

*Evidence:* `_foreign_test.py::test_a_foreign_transform_declares_its_struct`.

**claim: callbacks-batch-only.** Callback results can depend on the invocation's rows.
Such callbacks are batch-only and cannot compile through Confit.
The SQL check cannot establish a relation callback's semantics.
Relation callbacks can depend on their Arrow invocation and remain batch-only, including when referenced by a projection.

*Evidence:* `_projection_test.py::test_a_foreign_leaf_serves_in_batch_but_refuses_to_compile_by_name`.

```pycon
>>> import pyarrow as pa
>>> import pyarrow.compute as pc
>>> from sql_transform import SQLTransform, Transform, run
>>> def fit_mean(group):
...     return pc.mean(group["v"]).as_py()
>>> def subtract_mean(mean, rows):
...     return pa.table({"v": pc.subtract(rows["v"], mean)})
>>> center = Transform(fit_mean, subtract_mean, takes=("v",), returns=("v",))
>>> program = SQLTransform('''
... WITH p AS (SELECT center_fit(struct_pack(v := v)) AS theta FROM __FIT__)
... SELECT center_transform(p.theta, struct_pack(v := t.v)).v AS z
... FROM __THIS__ t, p
... ''', captured={"center": center})
>>> fit_data = pa.table({"v": [10.0, 20.0]})
>>> requests = pa.table({"v": [12.0, 22.0]})
>>> fitted = program.fit(fit_data)
>>> assert fitted.transform(requests).to_pydict() == {"z": [-3.0, 7.0]}
>>> assert run(program, fit_data).to_pydict() == {"z": [-5.0, 5.0]}

```

## Handle representations

**claim: handle-representations.** The generic callback protocol uses an opaque `STRUCT(type, id)` value.
`type` identifies the callback, and `id` selects its fitted instance.
Pure SQL members use a struct of fitted SQL values.
Raw projection estimators use BIGINT IDs.
These representations are separate protocols, not authenticated handles.

*Evidence:* `_foreign_test.py::test_theta_is_an_opaque_handle`, `_leaf_test.py::test_theta_is_data_in_the_params_table`, `_raw_test.py::test_scaler_canonical_cte_and_window_have_the_same_public_artifact`.
