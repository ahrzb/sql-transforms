# Decorrelation, and what of it survives being shipped as data

Date: 2026-08-08. A survey, not a design — the question is whether the
correlated `__FIT__` subquery can become a pre-aggregated params table, and at
what cost in refusals.

Eight readings: the classical lineage, modern unnesting, DuckDB's own plans,
the semantic traps, the materialization literature, this repo, plus a synthesis
and an adversarial critic. Every DuckDB result below is 1.5.5.

**The four load-bearing claims were re-run by hand before this was written.**
They are marked *(verified here)*; everything else is relayed and labelled.

---

## The frame

The shape is Kim's type-JA:

```sql
SELECT t.cat, t.price / (SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat) AS z
FROM __THIS__ t
```

Kim's 1982 `NEST-JA` is the one rule in the whole lineage whose temporary
relation names the **inner** relation alone:

```
Rt(C1..Cn, Cn+1) = SELECT C1..Cn, AGG(Cn+1) FROM R2 GROUP BY C1..Cn
```

That is exactly a params table. Everything published since that *fixes* Kim —
Ganski & Wong's outer join for the count bug, their θ-join, their DISTINCT,
Dayal's outerjoin-then-group, Neumann & Kemper's dependent join, SQL Server's
Apply — repairs him by pulling the **outer** relation into the temporary
relation. In an optimizer that is free. Here the outer relation is `__THIS__`
and does not exist at fit.

So the target is **"Kim NEST-JA verbatim, plus serve-side repairs"**, not
"decorrelation". Materializability is strictly narrower than soundness, and the
narrowing is precisely at *the correlation is a conjunction of equalities*.
Muralikrishna (VLDB 1992) states the condition in the form we need: predicates
`f1(R) = f2(S)` where each function references one relation only — a pure AST
check.

---

## What was measured, not assumed

### The `__THIS__`-only conjunct is the sharpest edge *(verified here)*

Ranked the highest risk, and it has **no counterpart in the classical
literature** — a plan rewrite has both relations in hand, so the case cannot
arise there.

```sql
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat AND t.region = 'EU')
```

```
oracle            [('a','EU',20.0), ('a','US',None)]
conjunct dropped  [('a','EU',20.0), ('a','US',20.0)]   <- a plausible number
CASE guard        [('a','EU',20.0), ('a','US',None)]   <- match
```

It is syntactically a textbook type-JA, it rewrites without complaint, and it
returns 20.0 where the answer is NULL. Every conjunct of the source `WHERE`
must land in exactly one of {params `WHERE`, join condition, serve `CASE`} —
a partition, asserted, not assumed.

### No aggregate whitelist can be correct *(verified here)*

Four strands each produced a *different* list of "aggregates that are non-NULL
on empty input". They are all wrong, because the category does not exist:

```
count(*)                              on empty -> 0
count(price) FILTER (WHERE price > 0) on empty -> 0
count_if(price > 0)                   on empty -> None     <- same count, other spelling
entropy(price)                        on empty -> 0.0
sum / avg / bool_or                   on empty -> None
```

Two spellings of one count disagree. The answer is a **zero-row probe** —
`SELECT <the subquery's whole select list> FROM __FIT__ WHERE false` — computed
at fit from `__FIT__`'s *schema* and none of its rows. It generalises to UDAFs
and to aggregates DuckDB has not shipped yet, and it cannot rot.

### `COALESCE` is unsound; a sentinel is not *(verified here)*

Three strands recommended `COALESCE(p.v, default)`. Two measured it wrong, and
none of the three noticed. With `CASE WHEN count(*)=0 THEN -1 ELSE max(f.price) END`
over a group whose `price` is entirely NULL:

```
probe (WHERE false) -> -1
oracle        [('a',None), ('b',5.0), ('zz',-1.0)]
COALESCE      [('a',-1.0), ('b',5.0), ('zz',-1.0)]   <- a hit turned into a miss
hit sentinel  [('a',None), ('b',5.0), ('zz',-1.0)]   <- match
```

`COALESCE` is safe iff the default is NULL *or* the expression can never be
NULL on a non-empty group. That is true for a bare `count(*)` and false for
every compound select list worth supporting. One `TRUE AS hit` column removes
a case analysis nobody can be trusted to redo per expression.

### The join predicate: mirror the operator, carry its null-rejection

The bug I measured before the survey — `=` correlation rewritten with
`IS NOT DISTINCT FROM` yields a value where the oracle yields NULL — has one
correct general form, and it is *not* "always use INDF" (that is P4, and P4 is
a **window** rule):

```sql
-- author wrote:  WHERE f.cat = t.cat
params: ... FROM __FIT__ f WHERE f.cat IS NOT NULL GROUP BY f.cat
serve : ... LEFT JOIN p ON p.k1 = t.cat

-- author wrote:  WHERE f.cat IS NOT DISTINCT FROM t.cat
params: ... FROM __FIT__ f GROUP BY f.cat          -- NULL group kept, it is reachable
serve : ... LEFT JOIN p ON p.k1 IS NOT DISTINCT FROM t.cat
```

Per conjunct, so composite and mixed-spelling keys compose unchanged. DuckDB's
own optimized plan for the `=` shape emits exactly this — `HASH_JOIN LEFT` over
`PERFECT_HASH_GROUP_BY` over `FILTER (cat IS NOT NULL)` — and the filter is
predicate-derived: present for `=`, absent for `IS NOT DISTINCT FROM`. Neumann
& Kemper state the same as a side condition (§3.3).

---

## Where the fleet refuted itself

Worth recording, because a survey whose strands all agree has checked nothing.

- **"Refuse inequality correlation" is wrong.** The classical strand concluded
  it from Ganski & Wong §5.3 (whose fix needs the outer relation) — cited, not
  checked. Four other strands independently reproduced `f.ts <= t.ts` exactly
  via a *prefix aggregate + `ASOF LEFT JOIN`*, including the strict and
  descending forms. The literature's silence is an artefact of optimizers
  having no reason to prefer a prefix scan. The honest objection to this class
  is params size, not soundness.
- **"Refuse on cross-type mismatch" over-refuses.** The fan-out is real
  (2 output rows for 1 input row on `VARCHAR '1'/'01'` vs `INTEGER 1`) but the
  fix is one line: group by the *comparison-typed* key, `GROUP BY cat::INTEGER`.
  Same for `COLLATE` — `GROUP BY cat COLLATE NOCASE` reproduces the oracle.
  The correct law is "the params key equivalence must **equal** the join
  predicate's equivalence, not merely refine it."
- **"Refuse an inner `GROUP BY`" over-refuses.** A two-level F-only
  pre-aggregate matches. The real condition is *the subquery returns one row
  per outer tuple*; the syntactic proxy is not it.

---

## The refusal list, with an honest judgement

Today this shape refuses **100%** of the time. Under the proposed rule the
adversarial corpus admits 22 of 32 shapes with zero unsound admits.

| Refuse | Judgement |
| --- | --- |
| Correlating conjunct under `OR` / `NOT` / `CASE`, or not `=`/`IS NOT DISTINCT FROM` | Inequality is **genuinely common** (rolling features, quantile transforms) and provably materializable — this refusal is temporary and its message must say so |
| A `__THIS__` reference in the subquery's select list | **Deliberately slightly broad.** `avg(f.price) - t.price` distributes; `avg(abs(f.price - t.price))` does not; measured `min(f.price * t.w)` giving −20.0 vs −10.0 for a negative weight. No cheap AST rule sits between them, and the workaround is one edit |
| Non-aggregated `__FIT__` leaf in the select list | Fine — DuckDB errors at execution anyway; this hoists it to construction |
| `LIMIT` / `ORDER BY` / window inside the subquery | Fine, and it kills the nastiest trap for free: the obvious `arg_max` params for that shape is silently wrong on a NULL payload |
| Order-sensitive aggregate without `ORDER BY`; `first`/`last`/`any_value`/`mode` | Fine — fit would freeze one arbitrary physical order and ship it as a model |
| Volatile function anywhere | Mildly broad. `current_date` in a recency window is real; better long-run answer is to freeze the clock into params as an inspectable constant |
| Nested correlated subquery; more than one relation in `FROM` | Fine for v1 |

**The denominator is missing.** The repo's corpus contains **zero** entries
with a correlated `__FIT__` subquery, so 31% measures one author's imagination.
Adding JA shapes to `_corpus_test.py` is the first gate, before arguing about
the rate.

---

## The two things a syntactic rule cannot see

### 1. A correct rewrite can be a total leak *(verified here)*

Every strand proposed `|params| / |F|` as the disclosure metric. One row
defeats it:

```
SELECT list(price) FROM __FIT__     -- |F| = 100,000
params rows = 1          ratio = 0.00001
values inside that row = 100,000    serialized = 888,890 bytes
```

The metric rates a 100% leak as maximally safe. And the strand that called the
uncorrelated `__FIT__` subquery "the single safest shape in the space" is the
same one — it is the shape that admits this. A budget must be on **serialized
bytes and per-cell cardinality**, never on rows.

Separately: a *syntactically perfect* rewrite on a near-unique key ships |F|
rows with the columns renamed (measured four ways, ratio 1.00). Soundness and
non-disclosure are independent properties, and folding disclosure into the
admission rule would both over-refuse valid transforms and under-protect the
training set. It belongs in DRAFT-20 as a declared budget, not an inferred
refusal.

### 2. The rewrite reproduces the *full-fit* encoder

`TargetEncoder(cv=2).fit_transform(X, y)` gives `[2.5, 2.5, 2.5, 2.5, 35, 35,
15, 15]`; `fit(X, y).transform(X)` gives `[…, 25, 25, 25, 25]`. sklearn
cross-fits precisely because a row's own target sits inside its own group mean.

A JA rewrite that is perfectly sound *as SQL* produces the second one. So
`t.fit(D).transform(D) == run(t, D)` — our faithfulness law — holds exactly
while the features are optimistically biased, and no rule in any strand fires.

This needs an explicit decision: either the system claims to be an ML
feature-engineering tool, in which case a keyed aggregate over `__FIT__` needs
a documented "not cross-fitted" statement and DRAFT-20 owns cross-fitting as a
params *kind*; or it is a SQL partial evaluator, in which case say so and stop
citing sklearn parity as evidence about leakage.

---

## What this does not solve

All training-set shipping funnels through `whole_fit()` in `_plan.py`. After
this rule:

1. **The bare cross join** `FROM __THIS__ t, __FIT__ f` — untouched. Needs an
   enumerated rewrite table (sum over affine, min/max over monotone *with a
   sign side-condition*, count invariant), measured to work for those and
   measured to have no general form.
2. **The `__FIT__`-side correlation fall-through** — **still ships all of F**,
   measured. Same mechanics apply but the fix is to hoist both `__FIT__`
   subtrees into one fit-time statement. Separate ticket.
3. **The recursive CTE** — untouched, and correctly so.
4. **`NOT IN` / `> ALL` / `> ANY` over `__FIT__`** — the most dangerous thing
   found outside the rule. DuckDB compiles them to a `MARK` join with an F-only
   right-hand side, so they *look* materializable, while both obvious rewrites
   are wrong in opposite directions on NULLs and empty groups. Needs its own
   audit or a named refusal; silence is the one option unavailable.

## Laws that would have to move

- **P4** ("params joins use `IS NOT DISTINCT FROM`, never `=`") is a window
  rule. Narrow it to PARTITION-BY-derived joins and add a correlation sibling.
- **P14** ("unseen group ⇒ NULL") is false for a JA aggregate — an unseen key
  takes the subquery's own empty-input value.
- **P3**'s multiplicity argument does not survive a coercing or collated
  predicate. Strengthen to *equal*, not *refine*.
- **P7** survives intact, and is why the type and qualification checks must be
  construction-time.

## Recommendation

Build it, scoped to the equality case, gated on a declared `__THIS__` catalog
(without one the type checks are undecidable and today's refusal stands, so it
is non-regressive). Three mechanical emissions — params query, join predicate,
serve-side `CASE` — plus the zero-row probe. No aggregate catalogue, no
distributivity table, no ASOF path, no size guard in v1.

Before any of it: **JA shapes in the corpus**, and the `len(out) == len(T)`
assertion in the serving path, which is one line and catches the whole fan-out
family from any source.
