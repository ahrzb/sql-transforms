# Decorrelation: a working reference

A technical reference for anyone building a rewrite that turns a correlated
subquery into a pre-computed table. Written for the `__FIT__` case but the
mechanics are the marginalizer's too — a window aggregate over the training
support and a correlated aggregate over it are the same object under different
syntax, and the same four decades of corrections apply to both.

Companion to `2026-08-08-decorrelation-survey.md`, which decides *whether* to
build the type-JA case. This one is the machinery.

All DuckDB results are 1.5.5, measured 2026-08-08.

---

## 1. Why our setting is not the literature's

Everything published on decorrelation optimises a **plan**: both relations are
present when the rewrite runs. We split the two halves across time and space —
the inner side is evaluated at fit against `__FIT__`, materialised, shipped, and
joined much later against a `__THIS__` nobody has seen.

That single difference decides which published rule is available:

| Rule | Temporary relation mentions | Available to us |
| --- | --- | --- |
| Kim's `NEST-JA` (1982) | inner only | **yes** |
| Ganski & Wong's outer-join count fix (1987 §5.2) | outer | no |
| Ganski & Wong's θ-join fix (1987 §5.3) | outer | no |
| Ganski & Wong's DISTINCT fix (1987 §5.4) | outer | no |
| Dayal's outerjoin-then-groupby (1987) | outer | no |
| Neumann & Kemper's `D`-join (2015 §3.2) | outer (`D := Π^dist(T)`) | no |
| Neumann & Kemper's **substitution** (2015 §4) | inner only | **yes** |
| SQL Server's Apply removal (2001, Fig. 4 (7)–(9)) | outer, and needs a key on it | no |

Two survivors, and they are the same rule found twice, thirty-three years apart.
Everything else repairs Kim by reaching into the outer relation, which we cannot
do. So **materialisability is strictly narrower than soundness**, and the
narrowing sits exactly at *the correlation is a conjunction of equalities*.

Kim states the precondition globally rather than per-rule — "For simplicity, the
op in a join predicate is assumed to be an equality operator in this paper"
(§2) — and Muralikrishna restates it in the form that is an AST check:

> For Kim's method to apply, only equi-join correlation predicates of the form
> `f1(R) = f2(S)` must be present. `f1` and `f2` are functions that reference
> only `R` and `S` respectively.
> — Muralikrishna, VLDB 1992, p. 92 n. 2

---

## 2. The classical algorithm

### Kim's taxonomy (TODS 7(3), 1982)

- **type-A** — uncorrelated, aggregate, scalar result. Evaluate once, substitute
  the constant. *(This is our already-working closed-subtree freeze.)*
- **type-N** — uncorrelated, non-aggregate, set result. Becomes a semi-join.
- **type-J** — correlated, non-aggregate. Becomes a join.
- **type-JA** — correlated, **aggregate**. The subject of this document.

### `NEST-JA`, verbatim

```
(1)  Rt(C1..Cn, Cn+1) = SELECT C1..Cn, AGG(Cn+1) FROM R2 GROUP BY C1..Cn
(2)  the inner block becomes
     SELECT Rt.Cn+1 FROM Rt WHERE Rt.C1 = R1.C1 AND ... AND Rt.Cn = R1.Cn
```

Kim adds: *"the primary key of Rt is its first n columns."* That is the
multiplicity guarantee — one row per key tuple — and it is why the serve-time
`LEFT JOIN` cannot duplicate rows of `T`. Measured: joining `T` to a params
table built by `GROUP BY` returns `|T|` rows; joining to a non-aggregated
per-row table returns more.

Step (1) is our params query. Step (2) is our serve-time join. No modification.

### The correction history

Each entry is a genuine bug in Kim, and each fix reaches somewhere we cannot.

| # | Bug | Found by | Fix | Reaches outer? |
| --- | --- | --- | --- | --- |
| 1 | **The count bug** — `COUNT` over an empty group must be 0; `Rt` never contains a row for a group with no matching tuples, so the row vanishes | **Kiessling, UC Berkeley memo, 1984** (not Ganski & Wong, who relay it) | outer join instead of inner | yes |
| 2 | After the outer join, `COUNT(*)` counts the null-extended row and returns 1 | Ganski & Wong §5.2.1 | rewrite to `COUNT(<inner column>)` | — |
| 3 | Non-equality correlation: `Rt` groups by *equal* key values, the query asks about a *range* of them | Ganski & Wong §5.3 | put the outer relation in `Rt` and rewrite the predicate to `=` | yes |
| 4 | Duplicates in the outer join column inflate `COUNT`/`SUM` | Ganski & Wong §5.4 | `SELECT DISTINCT` over the outer join column | yes |

Ganski & Wong quote Kiessling's own conclusion — *"there seems to be no general
way to recover values lost by COUNTs on a correlation level greater than 1"* —
and their diagnosis of the mechanism is exact:

> in the formation of the temporary relation, no tuples appear which do not
> match the predicates applied to the inner relation. Thus, the COUNT function
> will never return zero.

**Kiessling's pessimism does not bind at depth 1.** With an equality correlation
the lost value is recoverable entirely on the serve side, with no further
reference to `__FIT__` — see §5. That is the whole reason this case is tractable
for us where the 1980s literature was stuck.

Bug #4 has a measured wrinkle worth recording: Ganski & Wong assert that
duplicates affect `COUNT`, `AVG` **and** `SUM`. On their own published data,
`AVG` is unaffected — the duplication is uniform within a group, so it cancels.
`COUNT` and `SUM` inflate by the duplication factor.

---

## 3. The modern algebra

Neumann & Kemper (BTW 2015) unnest *any* query by introducing the outer side's
distinct correlation values as a relation:

```
T1 ⋈^D T2  ≡  T1 ⋈_≡ (D ⋈^D T2),    D := Π^dist_{F(T2) ∩ A(T1)}(T1)
```

with the stated intent: *"we first compute the domain D of all variable
bindings, evaluate T2 only once for every distinct variable binding."*

`A(T1)` is the outer relation's attribute set. For us the outer relation is
`__THIS__`, so **`D` is uncomputable at fit time**, and the entire chain that
begins with it is unavailable. Elhemali et al. (SIGMOD 2007 §1.3) confirm there
is no fallback: the alternatives are *navigational* (forward lookup invokes the
subquery one outer row at a time; reverse lookup starts with the subquery) and
every one of them needs `F` present at serve time.

But the paper's fourth step, **substitution**, is exactly our rewrite:

> `D ⋈^D Q ≡ Map(Q)` if D's attributes are in the equivalence class of
> attributes of Q. […] instead of joining with D, we can extend Q and compute
> the implied attribute value from D by using the equivalent attributes. Note
> that this only holds because D is a set.

And the paper is honest that it produces a superset pruned later — which is
precisely our params-then-serve-join split:

> substitution might increase the size of intermediate results, the relationship
> between the two formulations is not = but ⊇ … The tuples that do not have join
> partners will be eliminated by the reconstructing join.

### The chain, instantiated

For `T ⋈^D Γ_{;m:avg(f.price)}(σ_{f.cat = t.cat}(F))`:

```
1. introduce D    ≡ T ⋈_{t.cat ≡ d.cat} ( D ⋈^D Γ_{;m:avg}( σ_{f.cat=d.cat}(F) ) )
2. push through Γ ≡ T ⋈ Γ_{d.cat; m:avg}( D ⋈^D σ_{f.cat=d.cat}(F) )      -- needs D a set
3. push through σ ≡ T ⋈ Γ_{d.cat; m:avg}( D ⋈_{f.cat=d.cat} F )
4. SUBSTITUTION   ≡ T ⋈_{t.cat = p.cat} Γ_{f.cat; m:avg}(F)                -- D eliminated
```

Only step 4's *result* is buildable at fit. So the operational statement of the
admission rule is: **a correlated `__FIT__` subquery is fit-time unnestable iff,
after push-down, every free `__THIS__` attribute lies in a conjunctive-equality
equivalence class with an expression produced by the `F` subtree.**

Neumann's own hard example (Q2) is refused by the same test that refuses our
inequality case:

> here D cannot be eliminated, as there is a non-equi join with values from D,
> which prevents substitution … the domain of the outer query has to be
> transferred sideways.

### `D` is magic sets under another name

Bancilhon, Maier, Sagiv & Ullman (PODS 1986) rewrite rules so bottom-up
evaluation *"cuts down on the irrelevant facts that are generated"*; Seshadri et
al. (SIGMOD 1996) model magic *"as a special join method that can be added to
any cost-based query optimizer"* with cost formulas *"to decide whether it is
beneficial"*.

The consequence matters: restricting to `D` is an **optimisation**, not a
correctness device — unrestricted evaluation computes a *superset*. So dropping
magic, which is forced on us, is sound. It is *adding* magic that carries
conditions. Our params table is the unrestricted evaluation, and that is why it
is bigger than a plan-time temp relation would be. §7.

### Apply, and the key it needs

Galindo-Legaria & Joshi (SIGMOD 2001, Fig. 4) give the GroupBy-Apply identities:

```
(8)  R A^× (G_{A,F} E) = G_{A ∪ columns(R), F}(R A^× E)
(9)  R A^× (G^1_F E)   = G_{columns(R), F'}(R A^{LOJ} E)
```

with the footnote *"Identities 7 through 9 require that R contain a key R.key"*
and the remark that if `S` has no key *"one can always be manufactured during
execution."* Manufactured **during execution** — a fact about `T`, unavailable
to us. Neumann discharges the same obligation differently: `D` is duplicate-free
by construction. For us it is discharged a third way — the params table is a
`GROUP BY` result and is therefore key-unique on its grouping columns, which is
Kim's own remark from 1982.

Two more of their classes are worth knowing:

- **Class 2** (removable only by duplicating subexpressions) — they call the
  plan space *"additional research"*, and Elhemali warns *"the size of an
  expression can be increased exponentially."* For us duplication costs a
  **second params table**, which is free at serve time. Measured: a `UNION ALL`
  of two correlated branches reproduced exactly by two independent params tables.
- **Class 3** (exception subqueries) needs a runtime `Max1row` operator because
  multiplicity depends on data seen at execution. For us it is decidable at
  **fit**, since the inner side is fully materialised — one of the few places
  our setting is strictly *stronger* than an optimizer's.

---

## 4. What DuckDB actually does

Useful because DuckDB is our oracle, and because its optimizer implements
Neumann & Kemper.

The canonical JA shape:

```sql
SELECT t.cat, t.price / (SELECT avg(f.price) FROM f WHERE f.cat = t.cat) AS z FROM t
```

physical plan:

```
HASH_JOIN [Join Type=LEFT | Conditions=cat IS NOT DISTINCT FROM cat]
  L: SEQ_SCAN t
  R: PROJECTION > PERFECT_HASH_GROUP_BY [Groups=#0 | Aggregates=avg(#1)]
       > PROJECTION > FILTER > SEQ_SCAN f
```

with the `FILTER` node's `extra_info` verbatim `{"Expression": "(cat IS NOT NULL)"}`.

**That is our target shape, emitted by the oracle itself** — a standalone
`GROUP BY` over `F` joined to `T`, no `DELIM_SCAN` — and it carries the NULL
guard the naive rewrite forgets. The filter is **predicate-derived**: present for
`=`, absent when the correlation is written `IS NOT DISTINCT FROM`.

### `DELIM_JOIN` is not the gate

Plans containing `LEFT_DELIM_JOIN` + `DELIM_SCAN`: inequality correlation;
`count(*)` equality correlation; correlation on an expression
(`f.cat = lower(t.cat)`); equality correlation with an extra F-only predicate;
two-level correlation; `ORDER BY … LIMIT 1`.

**Every one of those was nonetheless reproduced exactly by a hand-written
params + serve-join rewrite.** Whether DuckDB materialises a delim scan is a
cost and physical-operator decision, not a materialisability verdict. Using its
presence as a construction-time gate would refuse at least six provably
correct shapes.

---

## 5. The empty-group value, and why no whitelist works

The count bug, restated for materialisation: a key present in `T` and absent
from `F` produces no params row, the `LEFT JOIN` misses, and the served value is
`NULL` — but the correlated original returns whatever the aggregate returns on
the empty input.

Empty-input values on 1.5.5:

```
count(*), count(x), count(DISTINCT x), approx_count_distinct, regr_count  -> 0
entropy(x)                                                                -> 0.0
sum, avg, min, max, list, string_agg, bool_and, bool_or, any_value,
stddev_*, var_*, product, bit_and/or/xor, histogram, arg_max, median,
quantile_cont                                                             -> NULL
```

Four independent readings each produced a **different** list of "aggregates that
are non-NULL on empty input", and all four are wrong, because the category is
not closed under spelling:

```
count(price) FILTER (WHERE price > 0)  on empty -> 0
count_if(price > 0)                    on empty -> NULL
```

Two spellings of one count, opposite answers. **Use a zero-row probe**, not a
table:

```sql
SELECT <the subquery's entire select list> FROM __FIT__ WHERE false
```

computed at fit from `__FIT__`'s *schema* and none of its rows. It generalises to
UDAFs and to aggregates DuckDB has not shipped, and it cannot rot. It also
survives a `HAVING`, which flips `count(*)`'s empty value from `0` to `NULL` and
therefore breaks any rule of the form "count needs `coalesce(·,0)`".

### Deliver it with a sentinel, never `COALESCE`

`COALESCE(p.v, default)` is sound iff the default is NULL **or** the expression
can never be NULL on a non-empty group. That is true of a bare `count(*)` and
false of every compound select list worth supporting. Measured, with
`CASE WHEN count(*)=0 THEN -1 ELSE max(f.price) END` over a group whose `price`
is entirely NULL:

```
oracle        [('a',None), ('b',5.0), ('zz',-1.0)]
COALESCE      [('a',-1.0), ('b',5.0), ('zz',-1.0)]   <- a hit turned into a miss
hit sentinel  [('a',None), ('b',5.0), ('zz',-1.0)]   <- match
```

One `TRUE AS hit` column in params, `CASE WHEN p.hit THEN p.v ELSE <probe> END`
at serve.

---

## 6. The rewrite catalogue

What each correlation shape becomes, and its status.

| Correlation | Fit-time object | Serve operator | Status |
| --- | --- | --- | --- |
| `f.k = t.k` (conjunction of equalities) | `GROUP BY e_F` + `WHERE e_F IS NOT NULL` | `LEFT JOIN ON p.k = e_T` | **exact**, measured |
| `f.k IS NOT DISTINCT FROM t.k` | `GROUP BY e_F`, NULL group **kept** | `LEFT JOIN ON p.k IS NOT DISTINCT FROM e_T` | **exact**, measured |
| mixed spellings, composite key | per-conjunct, each with its own guard | per-conjunct | **exact**, measured |
| `f.k = lower(t.k)` (expression on either side) | `GROUP BY f.k` | join on `lower(t.k) = p.k` | **exact**, measured |
| cross-type / collated key | `GROUP BY` the **comparison-typed** key (`cat::INTEGER`, `cat COLLATE NOCASE`) | as written | **exact**; grouping the raw column fans out |
| `f.ts <= t.ts` (order predicate) | prefix aggregate over `Π_ts(F)`: `sum(sum(x)) OVER (PARTITION BY k ORDER BY ts)` | `ASOF LEFT JOIN ON t.ts >= p.ts` | **exact**, measured — strict, non-strict and descending forms |
| aggregate linear in a `__THIS__` column, e.g. `avg(f.price - t.price)` | `(sum, count)` per key | `(p.s - p.n*t.price) / nullif(p.n,0)` | **exact**, measured |
| aggregate with a `__THIS__`-valued **threshold**, e.g. `count(*) WHERE f.price > t.price` | per-key ECDF | — | sound but O(\|F\|) — the training set, partitioned |
| disjunctive correlation | one params table per branch | inclusion–exclusion | additive aggregates only; loses bit-exactness |
| inner `GROUP BY` under an outer aggregate | two-level F-only pre-aggregate | as equality | **exact**, measured |
| `> ALL` / `> ANY` / `NOT IN` | `(max, count(*), count(col))` triple | — | both naive rewrites wrong in **opposite** directions on NULLs |

`> ALL` over ∅ is TRUE, `> ANY` over ∅ is FALSE, and a NULL member poisons the
result to unknown. Measured: `t.price > max(price)` gets two of three rows
wrong; the triple reproduces it. DuckDB compiles these to a `MARK` join with an
F-only right-hand side, so they **look** materialisable — this is the most
dangerous family in the space.

---

## 7. The materialisation theory

Three results carry over, and one famous one does not.

**Yan & Larson, "Eager Aggregation and Lazy Aggregation", VLDB 1995, Theorem 1**
— the eager-aggregation precondition. Two expressions are equivalent if the
aggregates are decomposable, the upper aggregates are of the right class, and
*"NGA_d → GA_d^J holds"* (the pushed-down grouping columns functionally
determine the pushed-down relation's join columns). Because we group **on** the
correlation key, condition (3) holds by construction, and the whole theorem
degenerates to a trichotomy over the conjuncts of the subquery's `WHERE`:

- pure-`__FIT__` → copy into the params `WHERE`
- pure-`__THIS__` → **serve-time `CASE` guard** (see §8, the worst trap)
- `e_F op e_T` with `op ∈ {=, IS NOT DISTINCT FROM}` → `GROUP BY e_F`, join on `e_T`

Anything else refuses, per conjunct.

**Halevy, "Answering queries using views: a survey", VLDB Journal 10(4), 2001**
— names our boundary in the field's own vocabulary. He distinguishes *equivalent*
rewritings (query optimisation: the rewrite must produce the same answer) from
*maximally-contained* rewritings (data integration: the best obtainable subset is
acceptable). **We are permanently in the first regime.** A maximally-contained
rewriting is literally this project's one unrecoverable outcome — a query that
builds and quietly computes something else. Goldstein & Larson (SIGMOD 2001)
give the practical matching conditions for views ending in a `GROUP BY`.

**Larson, "Data Reduction by Partial Preaggregation", ICDE 2002** — if params are
ever shared across grains, the rule is: you may preaggregate on any column set
that *functionally determines* the final grouping set, and every stored
aggregate must be decomposable. Note for later: DuckDB exposes **no mergeable
sketch state** — `approx_count_distinct`, `approx_quantile`,
`reservoir_quantile`, `median`, `mode` and `entropy` all return finished
scalars, with no HLL/t-digest/KLL type to store. Holistic aggregates therefore
cannot be re-grained at all.

**Gray et al.'s data cube classification — distributive / algebraic / holistic —
does not bind at the exact grain.** This is the most useful negative result in
the survey. Because the aggregate is *finished at the grain the correlation asks
for*, nothing is ever recombined:

```
params at grain (g,h) answering a query at grain (g):
  avg-of-avgs              51.0   vs true 34.67   WRONG
  sum(per-group count(DISTINCT))  4 vs true 3     WRONG
  median-of-medians        30.0   vs true 25.0    WRONG
at the exact grain: none of those operations occur — the stored value IS the answer
```

Measured to round-trip verbatim at the exact grain: `avg`, `sum`, `count(*)`,
`count(col)`, `count(DISTINCT)`, `min`, `max`, `stddev_samp`, `var`, `median`,
`quantile_cont`, `entropy`, `bool_and`/`bool_or`, `list`/`string_agg` (with
`ORDER BY`), `histogram`, and arbitrary UDAFs. **Storing `(sum, count)` instead
of `avg` buys nothing and costs bit-exactness** — measured, reassociated float64
sums land at `30.799999999999997` against DuckDB's `30.8`.

The only genuine non-admissions are aggregates whose value depends on physical
order: `string_agg`/`list`/`array_agg` without an explicit in-aggregate
`ORDER BY`, and `first`/`last`/`any_value`/`arbitrary`/`mode`. Measured: the
same multiset producing `'1.0,3.0,2.0'` and `'3.0,2.0,1.0'`. The objection is not
decomposability — it is that fit would freeze one arbitrary outcome and ship it
as if it were a model.

---

## 8. The trap catalogue

Each of these produces a plausible wrong answer, not an error.

**T1 — a pure-`__THIS__` conjunct dropped.** The sharpest edge in the whole
survey, and it has **no counterpart in the literature**, because a plan rewrite
has both relations in hand and the case cannot arise.

```sql
(SELECT avg(f.price) FROM __FIT__ f WHERE f.cat = t.cat AND t.region = 'EU')
```
```
oracle           [('a','EU',20.0), ('a','US',None)]
conjunct dropped [('a','EU',20.0), ('a','US',20.0)]
```

Gate: assert every conjunct of the source `WHERE` lands in exactly one of
{params `WHERE`, join condition, serve `CASE`} — a partition, checked.

**T2 — operator mismatch on NULL keys.** `=` correlation served with
`IS NOT DISTINCT FROM` matches the NULL group and yields a value where the
oracle yields NULL. Symmetrically, an `IS NOT DISTINCT FROM` correlation served
against `IS NOT NULL`-guarded params yields NULL where the oracle yields a value.
Mirror the author's operator; carry *its* null-rejection into params.

**T3 — `COALESCE` for the miss.** §5.

**T4 — coercion or collation makes the params grouping finer than the join's
equivalence.** The only trap that breaks `shape="map"` rather than a value:
measured **2 output rows for 1 input row** on `VARCHAR {'1','01'}` vs
`INTEGER 1`. Fan-out occurs iff the comparison's implicit coercion is *coarser*
than the `GROUP BY` equivalence. Fix by grouping the comparison-typed key.
Strengthen P3: the params key equivalence must **equal** the join predicate's
equivalence, not merely refine it.

**T5 — a non-unique params table.** `LEFT JOIN` to a non-aggregated per-row
table silently duplicates `T` (measured 3 rows out for 2 in). `GROUP BY`
guarantees uniqueness; a non-aggregate correlated lookup does not, and DuckDB
raises for it anyway (*"More than one row returned by a subquery used as an
expression"*).

**T6 — the two-sided range.** Nobody's rule covers it, and it is a shape people
write:

```sql
(SELECT sum(f.price) FROM __FIT__ f WHERE f.cat=t.cat AND f.ts<=t.ts AND f.ts>t.ts-3)
```

Prefix-differencing across two ASOF joins gets the in-range rows right and
returns `0.0` where the oracle returns `NULL` — **the count bug reappearing
where no join miss occurs at all**, so no `hit` flag fires. A parallel count
prefix gated on `n_hi - n_lo = 0` reproduces it. Sliding-window `max` is not
recoverable from prefixes at all.

**T7 — `ORDER BY … LIMIT 1`.** The obvious params aggregate is `arg_max`, which
is **silently wrong on a NULL payload**; only `arg_max_nulls_last` — which
DuckDB's own plan uses — matches.

**T8 — serialization is part of the contract.** No analogue in the literature,
because a plan rewrite never crosses a storage boundary and ours always does. A
DOUBLE key round-trips bit-exactly through parquet and DuckDB's JSON writer
(`0.1+0.2` written as `0.30000000000000004`, read back distinct from `0.3`,
`GROUP BY` still yielding three groups). But a **NaN key is written as the bare
token `NaN`**, which is not valid RFC 8259 and a strict reader rejects.

---

## 9. Disclosure

A *correct* rewrite can ship the entire training set. This is orthogonal to
soundness and must not be folded into the admission rule — doing so would both
over-refuse valid transforms and under-protect `F`.

**Cardinality.** `|params| = ndv(key) ≤ |F|`, with equality iff the key is
unique in `F`. Measured across four keys on one fixture: 0.40, 0.80, 0.80,
**1.00** — the last being every training row shipped, merely relabelled.

**The row metric is not enough.** Measured on `|F| = 100,000`:

```
SELECT list(price) FROM __FIT__   ->  1 params row, ratio 0.00001,
                                      100,000 values inside it, 888,890 bytes
```

Budget on **serialized bytes** and per-cell cardinality, never on rows. Note the
uncorrelated `__FIT__` subquery — the shape most naturally called "safest" — is
exactly the one that admits this.

**Invertibility.** A prefix-aggregate params table with `|params| ≪ |F|`
**differences back into the training rows bit-for-bit**
(`s - lag(s,1,0) OVER (ORDER BY ts)`). Summary-ness is not a size property.

**Three axes, not one.** (a) *Selecting vs blending* — `min`, `max`, `mode`,
`any_value`, `arg_max`, `quantile_disc`, `median` (odd n) return a **verbatim
training datum** at any group size; `avg`/`sum`/`stddev` blend. (b) *Group size*
— n=1 makes `avg` the raw value; n=2 with `(min,max)` discloses both rows;
`(sum,count,min,max)` together is a reconstruction kit. (c) *Membership* —
`SELECT DISTINCT key FROM __FIT__` is a membership-inference oracle when the key
is a user id.

**And the one that is not about size at all.** A JA rewrite reproduces the
**full-fit** encoder, not the cross-fitted one. Measured, sklearn 1.9.0:
`TargetEncoder(cv=2).fit_transform(X,y)` → `[2.5,2.5,2.5,2.5, 35,35,15,15]`;
`fit(X,y).transform(X)` → `[…, 25,25,25,25]`. sklearn cross-fits precisely
because a row's own target sits inside its own group mean. Our faithfulness law
`t.fit(D).transform(D) == run(t,D)` holds *exactly* while the features are
optimistically biased. Also measured: sklearn's `TargetEncoder` treats NULL as
its own category (`IS NOT DISTINCT FROM` semantics, not SQL 3VL) and maps an
unseen category to the global mean, not to NULL — so on **both** divergence axes
the ML convention differs from correlated-subquery semantics. That is an
argument for exact equivalence by default with named opt-in policies, not for
guessing.

---

## Bibliography

- Kim, W. *On Optimizing an SQL-like Nested Query.* ACM TODS 7(3), 1982.
- Kiessling, W. UC Berkeley Memorandum, 1984 — the count bug.
- Ganski, R. & Wong, H. *Optimization of Nested SQL Queries Revisited.* SIGMOD 1987.
- Dayal, U. *Of Nests and Trees.* VLDB 1987.
- Bancilhon, Maier, Sagiv & Ullman. *Magic Sets and Other Strange Ways to Implement Logic Programs.* PODS 1986.
- Muralikrishna, M. *Improved Unnesting Algorithms for Join Aggregate SQL Queries.* VLDB 1992.
- Gray et al. *Data Cube: A Relational Aggregation Operator.* ICDE 1996.
- Yan, W. & Larson, P. *Eager Aggregation and Lazy Aggregation.* VLDB 1995.
- Seshadri et al. *Cost-Based Optimization for Magic: Algebra and Implementation.* SIGMOD 1996.
- Lenz, H. & Shoshani, A. *Summarizability in OLAP and Statistical Data Bases.* SSDBM 1997.
- Galindo-Legaria, C. & Joshi, M. *Orthogonal Optimization of Subqueries and Aggregation.* SIGMOD 2001.
- Goldstein, J. & Larson, P. *Optimizing Queries Using Materialized Views.* SIGMOD 2001.
- Halevy, A. *Answering Queries Using Views: A Survey.* VLDB Journal 10(4), 2001.
- Larson, P. *Data Reduction by Partial Preaggregation.* ICDE 2002.
- Elhemali, Galindo-Legaria, Grabs & Joshi. *Execution Strategies for SQL Subqueries.* SIGMOD 2007.
- Neumann, T. & Kemper, A. *Unnesting Arbitrary Queries.* BTW 2015.
