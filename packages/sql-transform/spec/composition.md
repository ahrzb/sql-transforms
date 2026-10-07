# Composition and replay

## Members

**claim: member-arguments.** Members accept two relation arguments in SQL: `member(fit_relation, request_relation)`.
Members can contain other members, up to the public `MAX_DEPTH` limit.
`MAX_DEPTH` is the exported limit on nested member composition.

*Evidence:* `_calling_test.py::test_a_member_splices_and_matches_the_reference`.

**claim: captured-scope-resolution.** Composition resolves names within each member's captured scope.
An outer SQL name does not rebind a member's captured object.

*Evidence:* `_calling_test.py::test_splicing_is_capture_free`.

## Chaining

**claim: explicit-chaining.** Chaining states both fit and request arguments explicitly.
For example, `b(a(__FIT__, __FIT__), a(__FIT__, __THIS__))` fits `b` on `a`'s transformed fit data.

*Evidence:* `_calling_test.py::test_chaining_fits_on_transformed_data`.

**claim: projection-members.** Pure SQL projection members also support scalar fit/transform composition with a struct of fitted values.
Explicit keyed projection members preserve the author's join equality.

*Evidence:* `_leaf_test.py::test_theta_is_data_in_the_params_table`.

Marginalized window projections are not members ([claim: refuses-marginalized-projection-members](unsupported-forms.md)).

## Direct execution with run

**claim: run-binds-both-arguments.** `run(transform, data)` binds both arguments to `data` and executes the resolved program directly.
It does not fit or freeze first.
This provides the reference for comparing a general transform's fit-data execution with its fitted execution.

*Evidence:* `_calling_test.py::test_per_group_is_faithful_under_run`.

**claim: raw-estimators-not-run.** Raw projection estimators do not support this direct `run` path.
Compare them with independent estimator execution instead.

*Evidence:* `_resolution_test.py::test_raw_replay_retains_only_author_captures_and_fits_one_scope_once`.

```pycon
>>> import pyarrow as pa
>>> from sql_transform import SQLTransform, run
>>> fit_data = pa.table({"v": [10.0, 20.0]})
>>> requests = pa.table({"v": [12.0, 22.0]})
>>> shift = SQLTransform("SELECT t.v - p.lo AS v FROM __THIS__ t, (SELECT min(v) AS lo FROM __FIT__) p")
>>> scale = SQLTransform("SELECT t.v / p.m AS z FROM __THIS__ t, (SELECT avg(v) AS m FROM __FIT__) p")
>>> composed = SQLTransform("SELECT * FROM scale(shift(__FIT__, __FIT__), shift(__FIT__, __THIS__)) s", captured={"shift": shift, "scale": scale})
>>> fitted = composed.fit(fit_data)
>>> assert fitted.transform(requests).to_pydict() == {"z": [0.4, 2.4]}
>>> assert run(composed, fit_data).to_pydict() == {"z": [0.0, 2.0]}
>>> assert fitted.transform(fit_data).equals(run(composed, fit_data))

```

## Names and captures

**claim: caller-name-capture.** The constructor captures referenced caller names without retaining the caller frame.
Explicit `captured` entries win over caller names.

*Evidence:* `_calling_test.py::test_splicing_is_capture_free`, `_lifecycle_test.py::test_a_transform_built_after_a_fit_still_captures_from_its_own_frame`.

**claim: captured-mapping-adopted.** The constructor adopts the captured mapping so sklearn cloning preserves its identity.
Private per-scope estimator bindings do not add author capture keys.
Replay regenerates them.

*Evidence:* `_raw_test.py::test_source_replay_and_clones_do_not_add_generated_capture_keys`, `_sklearn_test.py::test_clone_keeps_names_that_the_frame_can_no_longer_resolve`.

## Source and replay

**claim: source-is-replay-input.** Constructor `source` is author-level explicit SQL.
Replay or clone with `source` and `captured`.
The constructor's `sql` is resolved diagnostic SQL, not replay input.
In particular, lowered list-fit calls must not pass through parsing as authored fit calls again.
For marginalization, `source` contains the explicit SQL produced by the derivation.

*Evidence:* `_boundary_test.py::test_clone_still_carries_captured_objects`, `_raw_test.py::test_source_replay_and_clones_do_not_add_generated_capture_keys`.
