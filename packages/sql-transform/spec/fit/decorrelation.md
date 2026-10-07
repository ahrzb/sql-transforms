# Decorrelation

## What decorrelation rewrites

**claim: decorrelation-runs-first.** It runs before normal freezing of independent fit subqueries.
Supported correlation lifting precedes the nested-query visit.

*Evidence:* `_correlate_test.py::test_a_cte_inside_the_subquery_is_lifted_like_any_other_spelling`, `_correlate_test.py::test_the_correlation_may_reach_a_fit_only_relation_instead`.

**claim: decorrelation-predicate-partition.** Decorrelation rewrites supported correlated fit subqueries into fit queries and queries over params tables.
The fit-only predicates stay in the fit query.
Cross-relation equalities become keys.
Request-only predicates stay in the params-reading query.

*Evidence:* `_correlate_test.py::test_a_fit_only_conjunct_stays_in_the_params_query`, `_correlate_test.py::test_a_this_only_conjunct_becomes_a_guard_not_a_dropped_predicate`.

## Keys, misses and guards

**claim: key-operator-preserved.** The pass preserves each authored `=` or `IS NOT DISTINCT FROM` operator.

*Evidence:* `_correlate_test.py::test_the_join_predicate_mirrors_the_operator_the_author_wrote`.

**claim: misses-and-guards-preserved.** It distinguishes a missing group from a present group whose result is NULL.
It preserves the authored aggregate's empty-input value and false request guards.
These cases do not share one universal NULL or `COALESCE` rule.

*Evidence:* `_correlate_test.py::test_an_unseen_key_takes_the_subquerys_own_empty_value`, `_correlate_test.py::test_a_false_guard_means_the_group_is_empty_not_that_the_answer_is_null`, `_correlate_test.py::test_a_present_group_whose_value_is_null_is_not_a_miss`.

### Example: guarded means and counts

This example uses guarded means and counts.
Batch execution preserves unseen keys, NULL keys, and false request guards.
Confit separately refuses this emitted correlated expression.

```pycon
>>> import pyarrow as pa
>>> from sql_transform import SQLProjection
>>> fit_data = pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0], "enabled": [True, True, True]})
>>> requests = pa.table({"g": ["NEW", None, "a", "a"], "v": [2.0, 14.0, 12.0, 12.0], "enabled": [True, True, True, False]})
>>> authored = SQLProjection('''
... SELECT t.g,
...        t.v - (SELECT avg(f.v) FROM __FIT__ f
...               WHERE f.g IS NOT DISTINCT FROM t.g AND t.enabled) AS d,
...        (SELECT count(*) FROM __FIT__ f
...         WHERE f.g IS NOT DISTINCT FROM t.g AND t.enabled) AS n,
...        (SELECT count(*) FROM __FIT__ f
...         WHERE f.g = t.g AND t.enabled) AS n_eq
... FROM __THIS__ t
... ''')
>>> fitted = authored.fit(fit_data)
>>> answer = fitted.transform(requests).to_pydict()
>>> assert answer == {"g": ["NEW", None, "a", "a"], "d": [None, 7.0, -3.0, None], "n": [0, 1, 2, 0], "n_eq": [0, 0, 2, 0]}
>>> try:
...     fitted.compile()
... except ValueError as refusal:
...     assert str(refusal).startswith("unsupported:")
... else:
...     raise AssertionError("this correlated expression requires a Confit refusal")

```

## Supported shapes

**claim: supported-correlation-lifting.** Supported lifting includes aggregate scalar subqueries and correlated derived tables produced by member composition.
The pass preserves column aliases when flattening those tables.

- the plain type-JA, `WHERE f.cat = t.cat`, with any aggregate including UDAFs
  and `list`;
- composite keys, mixing `=` and `IS NOT DISTINCT FROM` per conjunct;
- a `__THIS__`-only conjunct alongside the correlation, which becomes part of
  the lookup rather than being dropped;
- `__FIT__`-only conjuncts, which stay in the params query and shrink it;
- a correlation into a `__FIT__`-only relation rather than into `__THIS__` —
  including the per-group `LATERAL` form, where the correlating
  predicate arrives a level below the aggregate and is flattened first;
- the subquery in a select list, a `WHERE`, a `HAVING`, or under an enclosing
  `GROUP BY`: only the subquery's body is rewritten, so the enclosing query is
  untouched by construction.

*Evidence:* `_correlate_test.py::test_the_canonical_shape_is_no_longer_refused`, `_correlate_test.py::test_a_cte_inside_the_subquery_is_lifted_like_any_other_spelling`, `_correlate_test.py::test_the_correlation_may_reach_a_fit_only_relation_instead`.

Fit checks for columns that nested relations shadow, and it names each unsupported correlation: see [decorrelation refusals](decorrelation-refusals.md). Inequality, prefix and ASOF correlations refuse.
