# Unsupported forms

The forms below are not admitted.
Explicit alternatives preserve computations without making unsupported syntax legal.
Removed interfaces are in the [changelog](../records/changelog.md).

## Window projection chains

**claim: refuses-window-projection-chains.** Window projection chains refuse. State the fit and request stages explicitly, or compose general `SQLTransform` members.

An explicit projected fit source needs no marginal chain walker.
This grouped example states its fit computation and request computation independently.
It does not claim the grouped reduction reproduces every original window's reduction order.

```pycon
>>> import pyarrow as pa
>>> from sql_transform import SQLProjection
>>> fit_data = pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0]})
>>> requests = pa.table({"g": ["NEW", None, "a"], "v": [2.0, 14.0, 12.0]})
>>> fitted = SQLProjection('''
... WITH p AS (
...   SELECT g, avg(x) AS m
...   FROM (SELECT g, 2 * v AS x FROM __FIT__) f GROUP BY g
... )
... SELECT t.g, 2 * t.v - p.m AS d
... FROM __THIS__ t LEFT JOIN p ON t.g IS NOT DISTINCT FROM p.g
... ''').fit(fit_data)
>>> assert fitted.transform(requests).to_pydict() == {"g": ["NEW", None, "a"], "d": [None, 14.0, -6.0]}

```

*Evidence:* `_marginal_test.py::test_refusals_fire_pre_rewrite_in_the_authors_vocabulary`, `_marginal_projection_test.py::test_cte_chain_flattening`.

## Schema expansion

**claim: refuses-schema-expansion.** COLUMNS and star EXCLUDE, REPLACE and RENAME expansion are not admitted. Select explicit columns and aliases. Normal binding checks still apply.

*Evidence:* `_marginal_test.py::test_refusals_fire_pre_rewrite_in_the_authors_vocabulary`.

## Alias expansion into fit expressions

**claim: refuses-alias-expansion-into-fit.** A fit expression cannot read a sibling output alias.
Repeat the expression or state an explicit stage instead.
Input names that start with `_` are ordinary columns.

*Evidence:* `_marginal_test.py::test_refusals_fire_pre_rewrite_in_the_authors_vocabulary`.

## Standalone fit windows

**claim: refuses-standalone-fit-windows.** Standalone raw fit-window output, including a parked `_th` alias, refuses.
Use inline fit and application, `x_transform(x_fit(bundle) OVER (...), bundle)`, or a canonical explicit params source.

*Evidence:* `_raw_test.py::test_parked_raw_fit_window_names_the_inline_and_explicit_forms`.

## Marginalized projection members

**claim: refuses-marginalized-projection-members.** Marginalized window projections are not scalar or keyed projection members.
Use an explicit grouped SQL projection member instead.

*Evidence:* `_marginal_test.py::test_projection_scope_refusals_in_the_authors_vocabulary`, `_marginal_test.py::test_a_keyed_fit_scope_outside_its_transform_refuses_by_name`.

## Noncanonical raw fit sources

**claim: refuses-noncanonical-raw-fit-sources.** Shared-CTE, deep or joined raw fit sources refuse, and so does intermediate ID passthrough. Use direct `__FIT__` or one inline projection of `__FIT__`, then a direct request-level ID ([claim: canonical-raw-params-query](python/estimators.md)).

*Evidence:* `_raw_test.py::test_raw_ids_and_sources_have_one_positive_grammar`.
