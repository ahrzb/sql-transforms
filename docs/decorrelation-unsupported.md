# Decorrelation: what refuses, and the plan for each

A correlated `__FIT__` subquery is lifted into a keyed table — Kim's
`NEST-JA`, emitted verbatim. This page is the other side of that: every shape
the rule does **not** claim, why, and what lifting it would take.

The goal is not zero refusals. It is that the list stays short, stays written
down, and stays confined to shapes people do not write. A refusal nobody can
enumerate is indistinguishable from a bug.

Background: [the survey](reports/2026-08-08-decorrelation-survey.md) for the
decision, [the reference](reports/decorrelation-reference.md) for the
algorithms and the traps. Implementation in
`packages/sql-transform/sql_transform/model/_correlate.py`.

---

## The metric

`CorrelatedFit.reason` is a string from `_correlate.REASONS`. That set *is*
the refusal list — there is no unnamed refusal — and
`test_every_refusal_reason_is_documented` fails if a reason has no row below.

```python
try:
    SQLTransform(sql)
except CorrelatedFit as refusal:
    print(refusal.reason)     # 'not-an-equality'
```

Two things are deliberately **not** measured as refusals, because they are
not refusals:

- **`WholeTrainingSet`** is its own error. It is not about correlation; see
  [Shipping the training set](#shipping-the-training-set) below.
- **Artifact size.** A rewrite can be perfectly sound and still retain
  everything — `SELECT list(price) FROM __FIT__` is one row holding all of it,
  and a syntactically perfect keyed table on a near-unique key ships `|F|`
  rows with the columns renamed. Soundness and non-disclosure are independent
  properties; folding one into the other would over-refuse valid transforms
  and under-protect the training set. A declared byte budget is future work
  (DRAFT-20), not an inferred refusal.

---

## The refusal list

### `not-an-equality`

```sql
(SELECT avg(f.price) FROM __FIT__ f WHERE f.ts <= t.ts)          -- inequality
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat OR f.ok) -- under OR
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = upper(t.c) AND f.x > t.y)
```

A `GROUP BY` reproduces exactly the equivalence classes of `=` and
`IS NOT DISTINCT FROM`. Nothing else. A correlating conjunct under `OR` or
`NOT` does not partition at all, so there is no key to group on.

**This is the one refusal that is not a weird case.** Rolling windows, quantile
transforms and as-of features all correlate by inequality, and the survey's
fleet was wrong to call it unsound — four strands independently reproduced
`f.ts <= t.ts` exactly, via a *prefix aggregate* plus an `ASOF LEFT JOIN`,
including the strict and descending forms. The literature is silent because no
optimizer has a reason to prefer a prefix scan; that silence is not evidence.

**The plan.** A second params *kind*, not a variant of this one:

```sql
-- fit: the prefix aggregate, one row per distinct fitted timestamp
SELECT ts AS __key_0, sum(price) OVER (ORDER BY ts) AS __value FROM __FIT__

-- serve
ASOF LEFT JOIN p ON p.__key_0 <= t.ts
```

It needs its own emptiness story, which is why it is a separate design: trap
T6 in the reference is the count bug reappearing where **no join miss occurs
at all**, so a `count(*) = 0` test never fires and a parallel count prefix is
required to detect it. Owned by DRAFT-21.

`OR` and `NOT` stay refused. Distributing them means a union of keyed tables,
one per disjunct, and the params size multiplies — worth doing only if a real
query asks for it.

### `outside-where`

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

**The plan.** Lift the `__THIS__`-dependent part out of the aggregate when the
aggregate is *distributive over the outer term* — `sum(f.x) + n * t.y`,
`avg(f.x) - t.y`, `count(*)`. That is a small enumerated table plus a sign
side-condition for `min`/`max`, and it should be built only against real
queries: the general form does not exist, and guessing which entries matter is
how the table grows wrong.

### `not-aggregated`

```sql
(SELECT f.price FROM __FIT__ f WHERE f.cat = t.cat)
```

Every column of the subquery must collapse, or the lookup is not one row per
key. DuckDB errors on this at execution anyway ("More than one row returned by
a subquery") whenever a category has two rows; refusing hoists that to
construction, where P7 wants it.

**No plan, and none needed.** A one-row-per-key subquery whose value is not an
aggregate is either already broken or is `LIMIT 1` in disguise — see
`modifier`.

### `modifier`

```sql
(SELECT f.price FROM __FIT__ f WHERE f.cat = t.cat ORDER BY f.ts DESC LIMIT 1)
(SELECT DISTINCT f.cat FROM __FIT__ f WHERE f.cat = t.cat)
```

`ORDER BY`, `LIMIT` or `DISTINCT` inside the subquery. Grouping a `LIMIT 1`
gives one row for the *whole* relation, not one per key — catastrophically
wrong rather than slightly wrong, which is why it is checked rather than
trusted.

**The plan.** `ORDER BY … LIMIT 1` per key is `arg_max`, and the params query
writes itself:

```sql
SELECT cat AS __key_0, arg_max(price, ts) AS __value FROM __FIT__ GROUP BY cat
```

The trap is that this is *not* equivalent on a NULL payload: `arg_max` skips
rows whose value is NULL, and `ORDER BY … LIMIT 1` does not. So the rewrite
needs `arg_max(struct_pack(v := price), ts)` or an equivalent NULL-preserving
carrier. Cheap, well understood, and blocked on nothing but a decision to do
it. `DISTINCT` is separate and mostly pointless here.

### `grouping`

```sql
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat GROUP BY f.ok)
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat HAVING count(*) > 3)
```

The subquery does its own `GROUP BY`, `HAVING`, `QUALIFY`, or `GROUP BY ALL`.

**Over-refuses, and knowingly.** The survey found that a two-level `__FIT__`-
only pre-aggregate matches the oracle exactly; the real condition is *the
subquery returns one row per outer tuple*, and "has a `GROUP BY`" is a
syntactic proxy that is not it.

**The plan.** Append the correlation key to the existing `GROUP BY` and keep
the author's grouping underneath, as a nested aggregate. `HAVING` is a filter
on the inner groups and moves in unchanged. The reason this is not done yet is
that the one-row-per-key argument stops being structural and has to be
re-established for the two-level case.

### `window`

```sql
(SELECT max(avg(f.price) OVER (PARTITION BY f.ok)) FROM __FIT__ f WHERE f.cat = t.cat)
```

A window function inside the subquery. A window's frame is over the rows the
`WHERE` admitted, and grouping changes which rows those are.

**The plan.** Add the correlation key to the `PARTITION BY`. That is sound when
the key is already partition-constant, which the equality correlation makes
true — so this is closer to a missing case than a hard boundary. Left out of
v1 only because P4 (the window params law) and this rule would then overlap,
and the overlap needs its own thinking.

### `sample`

```sql
(SELECT avg(f.price) FROM __FIT__ f TABLESAMPLE 10% WHERE f.cat = t.cat)
```

Fit would freeze one draw and ship it as a model, and the two sides of
*freezing is faithful* would disagree by construction.

**No plan. This should stay refused.** Someone who wants a fixed sample can
write a seeded one and the refusal will not fire on it.

### `not-a-scalar-subquery`

```sql
WHERE EXISTS (SELECT 1 FROM __FIT__ f WHERE f.cat = t.cat)
WHERE t.cat IN (SELECT f.cat FROM __FIT__ f WHERE f.ok)
WHERE t.price > ALL (SELECT f.price FROM __FIT__ f WHERE f.cat = t.cat)
```

`EXISTS`, `IN`, `ANY`, `ALL`. DuckDB compiles all of them to a `MARK` join with
a `__FIT__`-only right-hand side, so they *look* materialisable — which is
exactly what makes them dangerous.

**This is the most dangerous shape outside the rule.** Both obvious rewrites
for `NOT IN` are wrong, in opposite directions, on NULLs and on empty groups.
Kim's own `NEST-N` handles `IN` by an assumption Ganski & Wong showed to be
false for duplicates.

**The plan.** Its own audit before any code, with the three-valued truth table
written out and gated on measured cases:

| shape | key insight |
| --- | --- |
| `EXISTS` | `count(*) > 0` per key — the easy one, and it is genuinely easy |
| `IN` | a keyed `bool_or`, but a NULL in the inner set turns false into NULL |
| `NOT IN` | the above negated is *not* the answer; empty set is TRUE, NULL-containing set is never FALSE |
| `> ALL` | `min`/`max` per key, plus the empty-set case, which is TRUE |

`EXISTS` is worth doing on its own. The rest should wait for the audit.

### `not-a-select`

```sql
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat
 UNION ALL SELECT 0)
```

A set operation rather than a plain `SELECT`. No plan: the shape is vanishingly
rare in a correlated position and the rewrite would have to key each arm and
re-union them.

---

## Shipping the training set

A separate error, `WholeTrainingSet`, for the paths where no keyed table
exists at all and honouring the text would mean every row in the artifact.

### A bare `FROM __FIT__` beside `__THIS__`

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

**The plan.** An enumerated rewrite table for the cases where the cross join
*does* reduce: `sum` over an affine expression, `min`/`max` over a monotone one
with a sign side-condition, `count` invariant. Measured to work for those and
measured to have no general form. Until then the one-edit refusal is the right
trade — it makes retention a thing the author wrote.

### A recursive CTE reading `__FIT__`

```sql
WITH RECURSIVE r(n) AS (SELECT count(*) FROM __FIT__ UNION ALL SELECT n-1 FROM r WHERE n > 0)
SELECT t.price, (SELECT max(n) FROM r) FROM __THIS__ t
```

A recursive CTE's self-reference is bound by the enclosing entry key, so
nothing inside the body can be hoisted out — hoisting any part leaves that name
dangling. `__FIT__` inside one used to be repointed at the whole training set:
the right answer, obtained by shipping every row to compute one number.

**The plan.** Freeze the body's `__FIT__`-only subtrees *in place*, by
reconstructing the `WITH RECURSIVE` wrapper around a rewritten body rather than
hoisting the body out. The subquery above is `(SELECT count(*) FROM __FIT__)`,
uncorrelated and F-only — an ordinary maximal freeze if the walk were allowed
to descend into a recursive body at all. The reason it is not is that the
descent has to prove it never lifts anything that references the self-name,
which is a check nobody has written.

---

## What is not on this list, and works

For contrast, since a page of refusals reads worse than the surface is:

- the plain type-JA, `WHERE f.cat = t.cat`, with any aggregate including UDAFs
  and `list`;
- composite keys, mixing `=` and `IS NOT DISTINCT FROM` per conjunct;
- a `__THIS__`-only conjunct alongside the correlation, which becomes part of
  the lookup rather than being dropped;
- `__FIT__`-only conjuncts, which stay in the params query and shrink it;
- a correlation into a `__FIT__`-only relation rather than into `__THIS__` —
  including the per-group `LATERAL` the guide teaches, where the correlating
  predicate arrives a level below the aggregate and is flattened first;
- the subquery in a select list, a `WHERE`, a `HAVING`, or under an enclosing
  `GROUP BY`: only the subquery's body is rewritten, so the enclosing query is
  untouched by construction.

## One hazard no AST check can see

A cross-type correlation key — fitted `VARCHAR` `'1'` and `'01'`, served
`INTEGER 1` — is two groups at fit and one key at serving, so the lookup
matches twice. Predicting it needs `__THIS__`'s types, which construction does
not have. It raises a named error at serving rather than answering quietly.

**The plan.** With a declared `__THIS__` catalog the comparison type is
decidable at construction, and the fix is to group by the *comparison-typed*
key — `GROUP BY cat::INTEGER`, and `GROUP BY cat COLLATE NOCASE` for the
collation case. The law is that the params key's equivalence must **equal** the
join predicate's, not merely refine it.
