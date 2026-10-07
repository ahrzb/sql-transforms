# Decorrelation refusals

This page lists each correlated shape that fit refuses, and why.

The goal is not zero refusals. It is that the list stays short, stays written
down, and stays confined to shapes people do not write. A refusal nobody can
enumerate is indistinguishable from a bug.

Background: the [decorrelation research](../../research/decorrelation/README.md) and the [type-JA survey](../../records/research/decorrelation/type-ja-survey.md). The code is `sql_transform/_correlate.py`.

## Named reasons

**claim: named-correlation-refusals.** `CorrelatedFit.reason` is a string from `_correlate.REASONS`. That set *is*
the refusal list — there is no unnamed refusal.

```pycon
>>> from sql_transform import CorrelatedFit, SQLTransform
>>> try:
...     SQLTransform("SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f WHERE f.ts <= t.ts) AS m FROM __THIS__ t")
... except CorrelatedFit as refusal:
...     print(refusal.reason)
not-an-equality

```

`WholeTrainingSet` and artifact size are not correlation refusals: see [freezing](freezing.md#the-whole-training-set).

*Evidence:* `_correlate_test.py::test_the_refusals_are_named_and_at_construction`, `_correlate_test.py::test_a_sample_on_the_table_reference_refuses_like_one_on_the_query`, `_correlate_test.py::test_a_nested_column_that_shadows_the_outer_alias_refuses`.

## The refusal list

### not-an-equality

**claim: refuses-not-an-equality.**

```sql
(SELECT avg(f.price) FROM __FIT__ f WHERE f.ts <= t.ts)          -- inequality
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat OR f.ok) -- under OR
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = upper(t.c) AND f.x > t.y)
```

A `GROUP BY` reproduces exactly the equivalence classes of `=` and
`IS NOT DISTINCT FROM`. Nothing else. A correlating conjunct under `OR` or
`NOT` does not partition at all, so there is no key to group on.

Rolling windows, quantile transforms and as-of features correlate by inequality, so this refusal meets common queries.

`OR` and `NOT` stay refused.

The refusal message names the hand-written form: a CTE over `__FIT__` that aggregates to one row for each lookup coordinate, joined to `__THIS__`.

*Evidence:* `_correlate_test.py::test_the_refusals_are_named_and_at_construction`, `_correlate_test.py::test_the_inequality_refusal_names_the_way_out`.

### outside-where

**claim: refuses-outside-where.**

```sql
(SELECT avg(f.price) - t.price   FROM __FIT__ f WHERE f.cat = t.cat)
(SELECT avg(abs(f.price - t.price)) FROM __FIT__ f WHERE f.cat = t.cat)
```

A `__THIS__` reference anywhere but the subquery's own `WHERE` — the select
list, the `FROM`, a `GROUP BY`.

**Deliberately slightly broad.** The first of those distributes: `avg(f.price)
- t.price` is a keyed mean minus a serving column. The second does not, and
neither does `min(f.price * t.w)` — measured, `-20.0` where the answer is
`-10.0`, for a negative weight. No cheap AST rule sits between them, and the
workaround is one edit: move the `__THIS__` term outside the subquery.

*Evidence:* `_correlate_test.py::test_the_refusals_are_named_and_at_construction`, `_correlate_test.py::test_flattening_never_changes_what_the_base_relation_is_called`.

### not-aggregated

**claim: refuses-not-aggregated.**

```sql
(SELECT f.price FROM __FIT__ f WHERE f.cat = t.cat)
```

Every column of the subquery must collapse, or the lookup is not one row per
key. DuckDB errors on this at execution anyway ("More than one row returned by
a subquery") whenever a category has two rows; refusing reports it at construction instead.

A one-row-per-key subquery whose value is not an aggregate is either already broken or a `LIMIT 1` in disguise: see [modifier](#modifier).

*Evidence:* `_correlate_test.py::test_the_refusals_are_named_and_at_construction`.

### modifier

**claim: refuses-modifier.**

```sql
(SELECT f.price FROM __FIT__ f WHERE f.cat = t.cat ORDER BY f.ts DESC LIMIT 1)
(SELECT DISTINCT f.cat FROM __FIT__ f WHERE f.cat = t.cat)
```

`ORDER BY`, `LIMIT` or `DISTINCT` inside the subquery. Grouping a `LIMIT 1`
gives one row for the *whole* relation, not one per key — catastrophically
wrong rather than slightly wrong, which is why it is checked rather than
trusted.

*Evidence:* `_correlate_test.py::test_the_refusals_are_named_and_at_construction`.

### grouping

**claim: refuses-grouping.**

```sql
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat GROUP BY f.ok)
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat HAVING count(*) > 3)
```

The subquery does its own `GROUP BY`, `HAVING`, `QUALIFY`, or `GROUP BY ALL`.

**Over-refuses, and knowingly.** The survey found that a two-level `__FIT__`-
only pre-aggregate matches the oracle exactly; the real condition is *the
subquery returns one row per outer tuple*, and "has a `GROUP BY`" is a
syntactic proxy that is not it.

*Evidence:* `_correlate_test.py::test_the_refusals_are_named_and_at_construction`.

### window

**claim: refuses-window.**

```sql
(SELECT max(avg(f.price) OVER (PARTITION BY f.ok)) FROM __FIT__ f WHERE f.cat = t.cat)
```

A window function inside the subquery. A window's frame is over the rows the
`WHERE` admitted, and grouping changes which rows those are.

*Evidence:* `_correlate_test.py::test_the_refusals_are_named_and_at_construction`.

### sample

**claim: refuses-sample.**

```sql
(SELECT avg(f.price) FROM __FIT__ f TABLESAMPLE 10% WHERE f.cat = t.cat)
```

Fit would freeze one draw and ship it as a model, and the two sides of
*freezing is faithful* would disagree by construction.

A seeded sample refuses too.

*Evidence:* `_correlate_test.py::test_the_refusals_are_named_and_at_construction`, `_correlate_test.py::test_a_sample_on_the_table_reference_refuses_like_one_on_the_query`.

### not-a-scalar-subquery

**claim: refuses-not-a-scalar-subquery.**

```sql
WHERE EXISTS (SELECT 1 FROM __FIT__ f WHERE f.cat = t.cat)
WHERE t.cat IN (SELECT f.cat FROM __FIT__ f WHERE f.cat = t.cat)
WHERE t.price > ALL (SELECT f.price FROM __FIT__ f WHERE f.cat = t.cat)
```

All three must be *correlated* to reach this. An uncorrelated
`t.cat IN (SELECT f.cat FROM __FIT__ f WHERE f.ok)` never reaches the rule at
all — it is an ordinary maximal freeze, one params row per qualifying training
row, and it answers correctly.

`EXISTS`, `IN`, `ANY`, `ALL`. DuckDB compiles all of them to a `MARK` join with
a `__FIT__`-only right-hand side, so they *look* materialisable — which is
exactly what makes them dangerous.

**This is the most dangerous shape outside the rule.** Both obvious rewrites
for `NOT IN` are wrong, in opposite directions, on NULLs and on empty groups.
Kim's own `NEST-N` handles `IN` by an assumption Ganski & Wong showed to be
false for duplicates.

*Evidence:* `_correlate_test.py::test_the_refusals_are_named_and_at_construction`.

### not-a-select

**claim: refuses-not-a-select.**

```sql
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat
 UNION ALL SELECT 0)
```

*Evidence:* not pinned by a test.

### not-in-a-subquery

**claim: refuses-not-in-a-subquery.**

```sql
SELECT t.cat, (WITH z AS (SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat)
               SELECT * FROM z) AS m
FROM __THIS__ t
```

The correlated subtree is a CTE definition or a set-operation arm rather than a
subquery, so there is no body to replace with a lookup — the rewrite works by
substituting the subquery's own `SELECT`, and here there isn't one in that
position.

*Evidence:* not pinned by a test.

### shadowed-by-a-nested-column

**claim: refuses-shadowed-by-a-nested-column.**

```sql
-- __FIT__ has a STRUCT column called `t`
SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat) AS m
FROM __THIS__ t
```

DuckDB binds `t.cat` to the nested column in preference to the outer relation,
so this is field access and the subquery is **not correlated at all**. Lifted
anyway, it builds a keyed table on a correlation nobody wrote and serves
plausible numbers. Measured: a `STRUCT` or `MAP` column wins, a plain column of
the same name does not, and a `STRUCT` without the field is a binder error — so
the test is on nestedness, and it runs at fit, where `__FIT__` first has a
schema.

This refusal happens at fit, not at construction, because it needs the schema of `__FIT__`.

*Evidence:* `_correlate_test.py::test_a_nested_column_that_shadows_the_outer_alias_refuses`, `_correlate_test.py::test_a_plain_column_of_the_same_name_is_not_shadowing`.

## Cross-type correlation keys

**claim: cross-type-key-error.** Fitted `VARCHAR` `'1'` and `'01'`, served `INTEGER 1`: two groups at fit and one
key at serving, so the lookup matches twice. Predicting it needs `__THIS__`'s
types, which construction does not have. The fitted lookup raises a named error when it runs, for example in batch `transform`, instead of answering quietly.

*Evidence:* `_correlate_test.py::test_a_cross_type_key_is_a_named_error_not_a_quiet_answer`.
