# Freezing

## Freezing order

**claim: maximal-fit-freeze.** Fit freezes maximal independent fit subqueries into params tables.
It does not recursively freeze smaller queries inside an already frozen subtree.

*Evidence:* `_freezing_test.py::test_freezing_is_complete`.

Decorrelation runs before this visit ([claim: decorrelation-runs-first](decorrelation.md)).

**claim: rewritten-cte-dependencies.** The planner records a common table expression's dependencies after rewriting its body.
Later reads therefore see the rewritten dependencies, not the original fit dependency.

*Evidence:* `_resolution_test.py::test_a_chain_of_ctes_propagates_what_it_reads`, `_resolution_test.py::test_a_subtree_reading_this_through_a_cte_is_not_frozen`.

## The static DISTINCT pick

**claim: static-distinct-pick.** One narrow exception also freezes a query with no remaining `__FIT__` read.
It must satisfy all these conditions:

- It is a flat ordinary `SELECT DISTINCT` of direct columns.
- Every selected column uses the same local source qualifier.
- It has one source and no other clauses, expressions, functions, or nested queries.
- Its source is an enclosing common table expression already rewritten to `SELECT * FROM` a prior fit-step params table.
- It has no outward references or duplicate output names.

A live source or a source that reads `__THIS__` does not qualify.
This rule lets window marginalization retain distinct keys and values instead of the full fit-time result table.

*Evidence:* `_freezing_test.py::test_a_distinct_pick_of_a_frozen_cte_ships_alone`, `_freezing_test.py::test_anything_but_the_flat_pick_stays_live`.

## Which params ship

**claim: reachable-params-only.** The planner executes each pick after its source.
It releases intermediates only after their last downstream fit use.
It removes dead common table expressions and ships only learned params reachable from the serving SQL.
If authored serving SQL still needs an explicitly retained whole-fit relation, that relation remains.

*Evidence:* `_freezing_test.py::test_a_dead_cte_is_neither_fit_nor_shipped`, `_freezing_test.py::test_a_distinct_pick_of_a_frozen_cte_ships_alone`.

## Artifact size

**claim: no-size-or-privacy-guarantee.** Fit does not guarantee a constant artifact size or privacy.
Near-unique keys and list-valued aggregates can retain much of the fit data.
A rewrite can be perfectly sound and still retain
everything — `SELECT list(price) FROM __FIT__` is one row holding all of it,
and a syntactically perfect keyed table on a near-unique key ships `|F|`
rows with the columns renamed. Soundness and non-disclosure are independent
properties; folding one into the other would over-refuse valid transforms
and under-protect the training set. Fit has no byte budget.
Inspect the artifact rather than infer a size or disclosure limit from decorrelation.

*Evidence:* `_freezing_test.py::test_params_are_measurable`.

## The whole training set

**claim: whole-training-set-refusal.** `WholeTrainingSet` names a refusal when an unreduced bare fit relation would remain in the artifact.
To retain fit data, write a subquery selecting the required rows and columns.
Use `(SELECT ... FROM __FIT__) f`.

`WholeTrainingSet` is a separate error, for the paths where no keyed table
exists at all and honouring the text would mean every row in the artifact.

### A bare fit relation beside the request table

```sql
SELECT t.price - f.price AS d FROM __THIS__ t, __FIT__ f
```

The rows really are needed — it is a cross product. What is refused is that
the size of the artifact would be a fact about freezing rather than about the
text. The fix is one edit, and it is also where you drop the columns you do
not need:

```sql
SELECT t.price - f.price AS d FROM __THIS__ t, (SELECT price FROM __FIT__) f
```

### A fit relation inside a recursive CTE

```sql
WITH RECURSIVE r(n) AS (SELECT count(*) FROM __FIT__ UNION ALL SELECT n-1 FROM r WHERE n > 0)
SELECT t.price, (SELECT max(n) FROM r) FROM __THIS__ t
```

A recursive CTE's self-reference is bound by the enclosing entry key, so
nothing inside the body can be hoisted out — hoisting any part leaves that name
dangling.

*Evidence:* `_correlate_test.py::test_a_bare_fit_beside_this_refuses_instead_of_shipping_the_training_set`, `_correlate_test.py::test_a_recursive_cte_over_fit_refuses_instead_of_shipping_it`.
