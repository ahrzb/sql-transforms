# The correlated `__FIT__` subquery, as a keyed table

Date: 2026-08-08. Follows
[the survey](../../reports/2026-08-08-decorrelation-survey.md) and
[the reference](../../reports/decorrelation-reference.md).

**The goal, in the words it was set in:** never ship the training set; keep the
minimal amount of data needed for the job, or refuse. Refusals are metered,
minimal and known, and confined to shapes people do not write.

---

## What decides, and what emits

Neumann & Kemper decide, Kim emits. They produce the same table; they differ in
what they make checkable.

```
Kim (1982)        the emission — exact SQL, and one row per key for free
Neumann (2015)    the decision — equivalence classes over conjunctive equalities
Yan & Larson      the proof obligation — cited in P3/P4, not implemented
```

Kim alone is a lookup table: shape in, rewrite out, and a query not in the
table can only be told "unsupported". Neumann's framing turns the same rule
into a *reason* — build the equivalence classes the join and filter conditions
induce, and a shape is admissible iff every free `__THIS__` attribute lands in
a class holding an `F`-side expression. That is an AST walk, and it produces
the refusal message.

Yan & Larson's Theorem 1 condition (pushed-down grouping columns functionally
determine the join columns) holds *by construction* once we group on the
correlation key, so implementing a check for it would verify a tautology. It
belongs in `properties.md` as the argument, not in the code.

## The partition

The whole algorithm. Every conjunct of the author's `WHERE` lands in exactly
one of three places:

```
reaches neither, or __FIT__ only     the params query's own WHERE
reaches __FIT__ and outward          a grouping key, if it is an equality
reaches outward only                 the lookup's WHERE
```

The third kind is the sharpest edge in the space and **has no counterpart in
the literature** — a plan rewrite holds both relations, so a predicate over the
outer one alone never needs moving.

```sql
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat AND t.region = 'EU')
```

```
oracle            [('a','EU',20.0), ('a','US',None)]
conjunct dropped  [('a','EU',20.0), ('a','US',20.0)]   <- a plausible number
```

It goes in the *lookup's* `WHERE`, not in a guard around it. A false one then
produces no rows, which is what the author's own filter produced: an empty
group, which takes the empty-input value. Guarding to NULL is right for `avg`
and wrong for `count`.

## The emission

```sql
-- source
SELECT t.cat, t.price / (SELECT avg(f.price) FROM __FIT__ f
                         WHERE f.cat = t.cat AND f.ok AND t.region = 'EU') AS z
FROM __THIS__ t

-- fit step __param_0: Kim's temporary relation
SELECT (f.cat) AS __key_0, (avg(f.price)) AS __value_0
FROM __FIT__ f WHERE f.ok AND (f.cat) IS NOT NULL
GROUP BY f.cat

-- fit step __param_1: the empty-input value, from F's schema and none of its rows
SELECT (avg(f.price)) AS __value_0 FROM __FIT__ f WHERE false

-- residual: only the subquery's body is replaced
SELECT t.cat, t.price / (
  SELECT CASE WHEN count(*) > 1 THEN error('fit and serving compare …')
              WHEN count(*) = 1 THEN any_value(__param_0.__value_0)
              ELSE (SELECT __value_0 FROM __param_1) END
  FROM __param_0
  WHERE __param_0.__key_0 = t.cat AND (t.region = 'EU')
) AS z
FROM __THIS__ t
```

Four decisions, each measured rather than reasoned:

**Only the subquery's body is replaced.** The enclosing `GROUP BY`, select list
and row count are untouched by construction — no restructuring of the outer
`FROM`, so `HAVING`, `ORDER BY`, nesting and `SELECT *` all keep working.

**A correlated aggregate, not a LEFT JOIN.** The readable emission anchors on a
one-row probe table and `LEFT JOIN`s the keyed one. DuckDB refuses it:
`NotImplementedException: Non-inner join on correlated columns not supported`.
Kim's own shape — a correlated aggregate with no `GROUP BY` — is what DuckDB
unnests natively, and it always returns exactly one row.

**Hit-ness is counted, never inferred from the value.** `COALESCE(v, empty)` is
unsound: a group that is present and whose value is legitimately NULL becomes a
miss. Measured, with `CASE WHEN count(*)=0 THEN -1 ELSE max(f.price) END` over
an all-NULL group. `count(*)` cannot be confused this way and needs no extra
column in params.

**A zero-row probe, not a list of aggregates.** Four survey strands each
produced a *different* list of "aggregates that are non-NULL on empty input".
The category does not exist: `count_if(x)` is NULL and
`count(x) FILTER (WHERE x)` is 0 — the same count spelled twice. The probe
generalises to UDAFs and to aggregates DuckDB has not shipped, and cannot rot.

## The join predicate mirrors the operator

Per conjunct, so composite and mixed-spelling keys compose unchanged.

```sql
-- author wrote:  f.cat = t.cat
params: … WHERE (f.cat) IS NOT NULL GROUP BY f.cat     -- NULL group unreachable
lookup: … WHERE p.__key_0 = t.cat

-- author wrote:  f.cat IS NOT DISTINCT FROM t.cat
params: … GROUP BY f.cat                               -- NULL group reachable, kept
lookup: … WHERE p.__key_0 IS NOT DISTINCT FROM t.cat
```

DuckDB's own optimized plan for the `=` shape emits exactly this —
`HASH_JOIN LEFT` over `PERFECT_HASH_GROUP_BY` over `FILTER (cat IS NOT NULL)`
— and the filter is predicate-derived: present for `=`, absent for
`IS NOT DISTINCT FROM`.

## Flattening, and why the guide's own pattern needed it

Splicing a member call writes the correlating predicate a level *below* the
aggregate:

```sql
z((FROM __FIT__ WHERE store = g.store), (FROM __THIS__ WHERE store = g.store))
-- becomes
(SELECT avg(price) AS m FROM (SELECT * FROM __FIT__ WHERE store = g.store) AS __FIT__) AS s
```

Without flattening the partition finds nothing to partition, and the shape
refuses for a reason that is an artefact of how it was written. A bare
`SELECT * FROM <base ref> WHERE p` derived table in the `FROM` — no projection,
no grouping, no modifiers — merges into its parent, and then it is an ordinary
type-JA.

This is why the rule claims **both** halves of DuckDB's `SUBQUERY`: the scalar
one in an expression, and the derived table in a `FROM`. The per-group
`LATERAL` the guide teaches is the second, and it used to ship the whole
training set. Measured on the guide's own data: 6 training rows became 5 param
rows, and with a million rows and three stores it is still 7.

## Nothing ships the training set unasked

`whole_fit()` is deleted. Two paths reached it, and both now raise
`WholeTrainingSet`:

- a bare `FROM __FIT__` beside `__THIS__` — the rows really are needed, but
  the artifact's size would be a fact about freezing rather than about the
  text. One edit fixes it, and it is also where you drop unused columns:
  `(SELECT price FROM __FIT__) f`;
- a recursive CTE reading `__FIT__` — nothing inside can be hoisted, because
  the self-reference is bound by the enclosing entry key.

Retention is still available. What is refused is retention nobody wrote down.

## Refusals are the metric

`CorrelatedFit.reason` is a string from `_correlate.REASONS`, and that set *is*
the refusal list — there is no unnamed refusal.
`test_every_refusal_reason_is_documented` fails if a reason has no row in
`docs/decorrelation-unsupported.md`, which carries the plan for each.

Nine reasons. One of them — `not-an-equality` — is not a weird case: inequality
correlation is common and provably materialisable via a prefix aggregate and
`ASOF LEFT JOIN`, and it is a separate params *kind* rather than a variant of
this one, because its emptiness story is different (no join miss occurs at
all, so no count fires).

## What this deliberately does not do

- **No disclosure rule.** A correct rewrite can be a total leak:
  `SELECT list(price) FROM __FIT__` is one params row holding all of F,
  888,890 bytes at 100k rows, and a keyed table on a near-unique key ships |F|
  rows with the columns renamed. Soundness and non-disclosure are independent;
  folding one into the other would over-refuse and under-protect. A declared
  byte budget is DRAFT-20.
- **No cross-fitting.** The rewrite reproduces the *full-fit* encoder.
  `TargetEncoder(cv=2).fit_transform` and `fit(...).transform(...)` differ, and
  a JA rewrite that is perfectly sound as SQL produces the second — so
  *freezing is faithful* holds exactly while the features are optimistically
  biased. That needs its own decision and its own params kind.
- **No `__THIS__` catalog.** Without one the comparison type of a correlation
  key is undecidable at construction, so a cross-type key raises a named error
  at serving instead. With a catalog the fix is `GROUP BY cat::INTEGER`.

## Laws that moved

- **P4** gained a scope note: it is a *window* rule, and a correlation key
  mirrors the author's operator instead. The general law both obey — the params
  key's equivalence must **equal** the join predicate's, not merely refine it —
  is also P3's real precondition.
- **P14** gained a scope note: an unseen key in a lifted correlation takes the
  subquery's empty-input value, not NULL.
- **P7** survives intact, and is why the shape checks are construction-time.
