# SQLProjection: a row-wise transform, servable one row at a time (v0)

Date: 2026-08-11. Status: designed with AmirHossein (this session).
Every DuckDB behaviour cited was measured on 2026-08-11 against the pinned
version, except the `LEFT JOIN` row-order claim under **Serving**, which is
carried over from the 2026-08-05 review round and marked where it appears.

## What it is for

`SQLTransform` is a table function `(F, T) -> R`. It is under no obligation to
return one row per input row, and it has no serving path — `Fitted.transform`
runs a batch through DuckDB and that is all there is.

Two things need more than that:

1. **A 1-1 serving pipeline.** One row in, one row out, through Confit's
   `DuckDBInferFn`, so a fitted artifact can answer a single request.
2. **A leaf inside another transform.** `p_fit` over `__FIT__` produces θ;
   `p_transform(θ, …)` over `__THIS__` produces the features. Today only a
   supplied Python pair (`_foreign.Transform`) can do this; an SQL-authored
   projection cannot.

*Leaf*, throughout: a transform's text can call other transforms, so they form
a tree, and a leaf is where it stops — a `(fit, transform)` pair with no SQL
inside for the model to descend into. Today every leaf is opaque Python and θ
is a pointer into a registry of fitted objects, which is what makes the second
item worth having: `_foreign.py` already promises that *"an SQL leaf gives an
inspectable params table; a fitted RandomForest gives a pointer."*

Both need the same guarantee, so they are one class.

## The model

> **A projection is a transform whose residual is row-wise: exactly one output
> row per `__THIS__` row, computed from that row and the params alone.**

It is not a new text and not a new parser. The same two-parameter text, the
same `_plan` freezing, the same `_correlate` lifting, the same refusals — plus
one gate at construction and one measurement at fit.

```sql
SELECT t.age - f.m AS d
FROM __THIS__ t
LEFT JOIN (SELECT country, avg(age) m FROM __FIT__ GROUP BY country) f
  USING (country)
```

That serves standalone, row at a time, and drops into a bigger transform as a
leaf. Nothing about the text says which; the text is a projection because of
what it does, and the class name is the author saying they intend it.

## Projection and transform are different kinds

They are siblings. Neither absorbs the other, and merging them is a category
error:

```
projection    a UDAF/UDF pair
              tfm_fit       : Agg[Struct<a,b,c>] => Struct<type, id>
              tfm_transform : (Θ, Struct<a,b,c>) => Struct<f1,f2>

transform     a table function
              (F, T) -> R,  under no row-wise obligation
```

The pair is the whole reason the row-wise guarantee has to be enforced rather
than hoped for: `_foreign._transform_batch` writes `out[position] = value` and
zips positions against values with `strict=True`. A leaf that changes
cardinality does not produce a wrong answer, it produces a `ValueError` from
inside a `zip`.

**And it is why a filtering projection is not a thing.** `tfm_transform` is a
*scalar UDF* — N values in, N values out, and no encoding for *no row here*. A
row it wanted to drop would have to come back as a NULL struct, which already
means something else (P14: an unseen group, the row stays and its output is
NULL). So dropping rows is not a feature a projection has not been given yet;
it is a feature the shape cannot hold. `WHERE` over a `__THIS__` column refuses,
permanently, and the refusal says why rather than promising a later flag.

Filtering is what a transform is for. `SQLTransform` may change cardinality
freely, because a table function is under no such obligation — and it does not
serve row-wise at all, so nothing in this design uses Confit's `shape="filter"`.

This mirrors `docs/specs/2026-08-05-fit-transform-split-design.md`,
which already states the pair and the one sugar over it.

## Marginalization is not the default

The existing `sql_transform.SQLProjection` reads `__THIS__` only and decides on
its own which window aggregates to freeze (`_marginalize`). That is what
produced the decorrelation cases: a `__THIS__`-side aggregate has to be turned
into a params table plus a join, and the correlations that fall out of doing it
automatically are the hard half of `_correlate`.

The new class does not do it. An aggregate or window over `__THIS__` refuses,
and the author writes the `__FIT__` half themselves:

```sql
-- refuses
SELECT age - avg(age) OVER (PARTITION BY country) AS d FROM __THIS__

-- what you write instead
SELECT t.age - f.m AS d
FROM __THIS__ t
LEFT JOIN (SELECT country, avg(age) m FROM __FIT__ GROUP BY country) f
  USING (country)
```

The trade is explicit: three more lines of SQL, and in exchange the two
relations are named, the retained rows are a subquery the author wrote, and no
rewrite has to guess.

Marginalization is not deleted, and it is not the default either. It comes back
later as an opt-in helper that *produces* a projection from a `__THIS__`-only
text:

```python
SQLProjection(sql)                 # you wrote the __FIT__ half
SQLProjection.marginalize(sql)     # later: derive it from a __THIS__-only text
```

Same class, same guarantees, same refusals — the helper is a rewrite in front
of the constructor, not a second mode inside it. Which is the point: whatever
it derives is a text you can read, and everything downstream of the constructor
cannot tell how the `__FIT__` half got there.

## What refuses

Two new refusals, and that is the entire budget.

### `NotRowWise`, at construction

Read off the residual — the text that survives freezing, where `__FIT__` is
already gone and only `__THIS__` and params tables remain:

The rule the table falls out of: **the spine — every level that carries the
batch's rows — must be a pure projection over row-keeping joins; a level that
reads only params is free**, because it is a constant table at serving. Ten
reasons, closed (`REASONS` in `_projection.py`; the test walks it for gaps):

| Shape | Why |
|---|---|
| aggregate on the spine | the answer depends on the other rows in the batch |
| window function on the spine | same |
| `GROUP BY` / `HAVING` | same |
| `DISTINCT` / `ORDER BY` / `LIMIT` (modifiers) | drops rows or destroys the row correspondence |
| `WHERE` / `QUALIFY` on the spine | drops rows, and a scalar UDF has no encoding for *no row here* |
| `__THIS__` entering the row stream twice | a self-join multiplies rows |
| a set operation over the batch | stacks the batch onto something else |
| a recursive CTE over the batch | iterates over the batch |
| a join that can drop or duplicate the batch's rows | keyed `INNER` drops on a miss; `LEFT` with the batch on the right drops it; `FULL` adds rows |
| off-spine (`spine`) | `__THIS__` read from an expression rather than FROM — a correlated subquery over the batch reads the batch's *other* rows even when it preserves cardinality; and a text that never reads `__THIS__` cannot track the batch at all |

The earlier draft of this section said a `WHERE` reading only params columns
was fine, "constant at serving". Implementation corrected it, twice over: a
joined params column varies per row through its key, and even a genuinely
constant false predicate zeroes every batch. Any spine `WHERE` refuses. The
free spelling — pinned by a test — is the `WHERE` *inside* the params
relation, where it filters training rows instead of serving rows.

Allowed joins on the spine, exactly: `LEFT` with the batch on the left (ASOF
included — it matches at most one row by its own semantics), `RIGHT` mirrored,
and the unconditional cross join, whose one-row obligation the fit-time check
owns.

Everything on that list is provably not row-wise, so the list is closed rather
than a matter of taste. It is also the list of things `SQLTransform` still
does; nothing is being taken away, it is being made a different class.

### `KeyNotUnique`, at fit

A join to a params table is row-wise only if it matches **at most one** row.
Nothing static can prove that, and the failure is silent. Measured:

```sql
WITH s AS (SELECT store, price AS m FROM __FIT__)
SELECT t.store, t.price / s.m AS z FROM __THIS__ t LEFT JOIN s USING (store)
```

```
params={'__param_s': 4}    1 row in -> 3 out   [z=0.333, z=0.5, z=1.0]
```

Four training rows in the artifact, three output rows for one input row, and no
error. The same text with `avg(price) … GROUP BY store` gives `params=2` and one
row out. Both build today under `SQLTransform`; both pass `_refuse_whole_fit`,
because each names its rows and columns in a subquery.

So the check is a measurement at fit, where the params tables exist. One query
per join, and it produces the refusal message rather than just a verdict:

```sql
SELECT k1, …, count(*) n FROM __param_s
GROUP BY k1, … HAVING count(*) > 1 ORDER BY n DESC LIMIT 1
```

A params table **cross-joined** (`FROM __THIS__ t, p` — no key at all) must
instead have exactly one row. That is not an edge case: `FROM __THIS__ t,
(SELECT avg(price) m FROM __FIT__) f` is the most common shape in the guide,
and it is row-wise precisely because the subquery returns one row. **Zero rows
refuses too**: a cross join to nothing deletes every serving row, more quietly
than multiplying them. And the probe orders by count then keys — two keys tied
on count made the refusal message flap between runs, measured the unpleasant
way. An `OR` in the join condition proves nothing about match counts and falls
back to the one-row rule; extra non-equality conjuncts pass, because they only
filter matches uniqueness already bounds at one.

The refusal names the offender, straight out of the query:

```
__param_s joins __THIS__ on (store), but store 'S1' has 3 rows, so one
serving row would become 3. Aggregate or de-duplicate it.
```

`GROUP BY` rather than `count(DISTINCT …)` for two measured reasons.
`count(DISTINCT k1, k2)` is a `BinderException` — the multi-column spelling is
`count(DISTINCT (k1, k2))`, which then yields a bare number with no offending
key in it. And `GROUP BY` folds NULL keys into one group, which is what the
join the model actually emits needs: `_correlate` lifts to `IS NOT DISTINCT
FROM`, never `=`, so NULL-keyed duplicates really do multiply. Measured, two
probe rows against a params table with two NULL-keyed rows:

```
=    join on NULL key -> 3 rows      # NULL-keyed duplicates never match, harmless
INDF join on NULL key -> 4 rows      # they match, and multiply
```

So the check is exact for a lifted lookup and conservative for an
author-written `=` join, where duplicate NULL keys would have been harmless.
That over-refusal is a genuinely weird case — a training relation with a NULL
join key, duplicated, reached by `=` — and it costs one line of SQL to be wrong
about in the safe direction.

Deliberately a measurement and not a syntax rule. A syntax rule ("the joined
relation must `GROUP BY` the join keys") is checkable at construction and
refuses two idioms that are correct:

```sql
SELECT DISTINCT store, tier FROM __FIT__
SELECT * FROM __FIT__ QUALIFY row_number() OVER (PARTITION BY store) = 1
```

A refusal you hit while writing correct SQL is not a weird case. Paying a
fit-time refusal to keep those working is the right side of the trade — and it
is the same P7 carve-out DRAFT-24 already took for learned output shapes.

## Serving

Two things, and only two. `transform` is the oracle — DuckDB, batch. `compile`
hands back a serving function and gets out of the way.

```python
p = SQLProjection(sql)
fitted = p.fit(TRAIN)

fitted.transform(X)              # pa.Table, num_rows == X.num_rows

fn = fitted.compile()            # confit.DuckDBInferFn
fn.infer({"age": 31})            # confit's own surface, not re-exported
fn.infer_rows(rows)
fn.infer_arrow(batch)
fn.backend, fn.boundary, fn.output_model, fn.shape
```

`compile()` builds the `DuckDBInferFn` from the same residual, the same params
tables and the same `shape="map"`, and returns it. It is not wrapped. The
existing `sql_transform.SQLProjection` wraps instead, and pays for it twice:
six delegating members (`infer`, `infer_batch`, `backend`, `boundary`,
`output_model`, and the lazily-cached `_serving_fn`), and a state coupling —
`fit` has to remember `self._fn = None`, because a refit silently invalidates a
prepared serving function. A `compile()` that returns a fresh object has no
such invariant to forget.

`shape="map"` is forced, not chosen: it is the same fact as `tfm_transform`
being a scalar UDF, seen from the serving side. The compiled function returns a
row per row, never `None`, and no caller downstream gains an optional case.

Confit's contract makes the compiled path and the DuckDB path bit-exact or
refuses by name.

The row path gets the residual respelled — no-ops to DuckDB, load-bearing to
Confit's stricter surface (all three measured 2026-08-12): freezing's own
passthroughs inline (`(SELECT * FROM __param_0) f` → `__param_0 AS f`; Confit's
FROM takes tables and joins, not derived tables); a cross join beside the
batch becomes `LEFT JOIN ... ON 1 = 1`, because Confit's map shape statically
refuses INNER and cannot know the fit probe measured the params side at one
row — and `ON TRUE` fails in the door, printing back as `CAST('t' AS
BOOLEAN)`; and the row model is trimmed to the columns the flattened residual
can read, because Confit requires every declared attribute on every input row,
so an unread label column must not be in the serving contract at all.

**Row order is threaded, not assumed.** A `LEFT JOIN` to a params table emits
unmatched rows last — measured in the 2026-08-05 review round, not here, and
`_projection._row_ordered_sql` exists for exactly this. The batch path appends a
`__cf_row` ordinal to
`__THIS__`, threads it through every spine `SELECT`, orders by it and drops it.
The construction gate guarantees every level is row-preserving, so the extra
select item is always lawful.

**The serving row model comes from the fit relation's schema.** No new
constructor parameter. Every field is `T | None` with a `None` default (as
`_model_from_arrow` already builds them), so a label column present in
`__FIT__` and absent at serving is harmless — the serving row simply does not
supply it. It breaks only when `__THIS__` carries a column `__FIT__` does not
have *and* the SQL reads it, which refuses at fit by name.

## The leaf role

A projection occupies the slot a `_foreign.Transform` occupies today. It does
**not** fill in that dataclass: it is spliced, and both halves become ordinary
SQL. The three decisions that shape this are stated as D1–D3 below.

What the author writes, and what it becomes:

```sql
-- written
SELECT t.k, sc_transform(sc_fit(struct_pack(v := t.v)) OVER (PARTITION BY t.k),
                         struct_pack(v := t.v)).v AS z
FROM __THIS__ t

-- spliced, for a projection whose params are single-row aggregates
SELECT t.k, (t.v - th.mean) / th.scale AS z
FROM (SELECT k, v, {'mean': avg(v)          OVER (PARTITION BY k),
                    'scale': stddev_pop(v)  OVER (PARTITION BY k)} AS th
      FROM __THIS__) t
```

Both forms were measured on 2026-08-11 and give the same values. Note the
syntax: `{…} OVER (…)` is a `ParserException`, so the aggregate distributes
over the struct's fields rather than wrapping it — each field carries its own
`OVER`.

Used against `__FIT__` rather than `__THIS__`, the same splice produces the fit
table, and `p_fit(…) OVER (PARTITION BY country)` becomes one θ per country:

```sql
SELECT k, {'mean': avg(v), 'scale': stddev_pop(v)} AS th FROM __FIT__ GROUP BY k
-- ('x', {'mean': 20.0, 'scale': 8.165}), ('y', {'mean': 200.0, 'scale': 100.0})
```

θ crosses joins **as a value** — it is a struct, not a registration — so the
guide's train/serve pattern works unchanged (implemented and gated, 2026-08-12):

```sql
SELECT t.store, p_transform(f.theta, struct_pack(price := t.price)).z AS z
FROM __THIS__ t
LEFT JOIN (SELECT store, p_fit(struct_pack(price := price)) AS theta
           FROM __FIT__ GROUP BY store) f
  ON t.store = f.store
-- the fit subquery freezes into a params table of readable θ structs,
-- {__param_0: {m, s}} per store; a join miss is a NULL θ whose every read
-- is NULL — P14 falls out of SQL instead of being implemented
```

## Decisions

### D1 — θ carries the parameters as data, not a pointer

**Status.** Proposed, 2026-08-11.

**Context.** `_foreign` makes θ a `Struct<type, id>` — an integer into a Python
registry of fitted objects. That makes `x_transform` a function of
`(θ, row, registry)` rather than of its arguments: the SQL cannot be shipped
without the Python, and nothing about θ says how big the fitted thing is.

**Decision.** For a projection leaf, θ is a struct of the learned parameters.
`Struct<type, id>` is retained only for leaves that are not SQL.

**Evidence.** Six transformers, measured — θ, then the transform half it admits:

| transformer | θ | `x_transform(θ, row)` |
|---|---|---|
| StandardScaler | `{mean: 20.0, scale: 8.165}` | `(v - θ.mean) / θ.scale` |
| MinMaxScaler | `{lo: 10.0, hi: 30.0}` | `(v - θ.lo) / nullif(θ.hi - θ.lo, 0)` |
| SimpleImputer | `{fill: 20.0}` | `coalesce(v, θ.fill)` |
| OrdinalEncoder | `{cats: ['a','b','c']}` | `list_position(θ.cats, c) - 1` |
| KBinsDiscretizer | `{edges: [16.6, 23.2]}` | `length(list_filter(θ.edges, e -> e <= v))` |
| TargetEncoder | `{m: {'a':1.0,'b':0.0,'c':0.0}}` | `θ.m[c]` |

None of those transform halves is a UDF. Each is a scalar SQL expression over
`(θ, row)` — which is to say, each is a projection.

**Consequences.** θ-as-data puts the artifact's size *in the text*, which is
the same principle `_refuse_whole_fit` already enforces for retention: a
`{cats: [...]}` holding 50k categories is visibly 50k, and a `TargetEncoder`
over a high-cardinality key is visibly close to shipping the training set. A
`Struct<type, id>` hides all of it behind an integer. Whether that visibility
should eventually become a refusal is not decided here.

Two θ shapes now coexist. They are not two spellings of one thing — they are
the two leaf kinds, and the pointer is the fallback rather than the design.

**What this does not solve.** θ's SQL type varies per projection and must be
known before the function is registered. D2 dissolves that: nothing is
registered.

### D2 — the pair is spliced, not registered

**Status.** Accepted by AmirHossein, 2026-08-11.

**Context.** `_foreign.Transform.register` creates two DuckDB functions per
leaf. That puts Python in the row path, which is exactly what a 1-1 serving
pipeline cannot afford.

**Decision.** A projection leaf is spliced into its host as SQL. No
`create_function`, no Python in the row path.

**Precedent.** This is already how member transforms compose, and the reason is
recorded in `_splice`'s own docstring: *"Splice, never emit a DuckDB macro:
measured, a table macro invoked under `LATERAL` does not see the correlation
and silently returns the whole-table answer for every group."*

**Consequences.** Confit can serve the whole pipeline with no Python callback.
Capture-avoiding substitution stops being incidental and becomes load-bearing —
see D3. And the `pa.Schema` widening loses its projection-side justification: a
spliced leaf declares no struct types at all. It survives on the Python-leaf
case alone (`from_estimator(OrdinalEncoder(), …)` dies at bind today), so it is
still worth doing but is **no longer a prerequisite** for this work.

### D3 — errors are attributed to the definition site

**Status.** Proposed, 2026-08-11, in response to D2's cost.

**Context.** A registered `sc_transform` attributes its own errors: the
function name is in the message, and `_Registry.keep` already carries a leaf's
own exception out through DuckDB's rewrapping so that *"a refusal keeps its
name"*. Splicing throws that away — after inlining there is one merged text,
and a binder error in it names nothing.

**Decision.** Two mechanisms, in this order.

1. **Refuse at the definition site.** `SQLProjection(sql)` parses, resolves,
   plans and refuses at its own construction, against its own text, before any
   host exists. Every refusal in this spec fires there. Only errors that come
   into existence *by merging* can reach the host.
2. **Name synthesized things after where they came from.** `_splice` already
   alpha-renames a member's free names to `{name}__{free}`, and `_reserve`
   keeps the `__` prefix clear so those cannot collide with an author's names.

**How it landed (2026-08-12).** Mechanism 2 shrank to nothing for projection
leaves, because the splice is a *substitution*: the leaf's aliases dissolve
into the host's bundle expressions and its params reads into `struct_extract`
over θ, so no name of the leaf's survives into the merged text — there is
nothing left to prefix. What remained is mechanism 1 plus its extension: the
leaf-shape refusals, which cannot fire at the projection's own construction
(they depend on how the host uses it), fire at the *host's* construction and
every one opens with the projection's name. The gate pins that: two deliberate
errors, each producing `"{stem} is a projection used as a leaf, and …"`.

**Consequences.** The residual unattributed class is DuckDB bind errors inside
substituted expressions — a VARCHAR bundle field reaching `avg`, say — which
carry DuckDB's message at fit. Loud, wrong-free, but host-attributed; living
with that is the recorded trade.

### Prerequisite: typed leaf structs

`_foreign._struct_sql` hardcodes `DOUBLE`, so a projection keyed on a string
cannot be a leaf. Measured:

```
struct_pack(cat := 'S1')    -> BinderException: No function matches ... 'leaf(STRUCT(cat VARCHAR))'
struct_pack(cat := '3.5')   -> BinderException      # even the numeric-looking string
struct_pack(cat := 7)       -> [(1.0,)]             # int widens, correctly
```

Loud rather than silent, and at bind time — but it makes the leaf role
numeric-only. `takes`/`returns` become `pa.Schema`:

```python
Transform(fit=..., transform=...,
          takes=pa.schema([("v", pa.float64())]),
          returns=pa.schema([("v", pa.float64())]))
```

`pa.Schema` rather than a `dict[str, pa.DataType]` because pyarrow already has
the vocabulary and a dict would be a second one. It also puts `Transform` on the
same footing as `_udf.PythonTransform`, which `_projection._fit_step` already
builds with `pa.schema(...)` / `pa.struct(...)`.

Cost: eight one-line edits — six tests, one doctest in
`docs/sql-transform-guide/02-what-the-two-parameter-form-changes.md`, one line
in `2026-08-07-datamodel-redesign-design.md`. None change behaviour;
`pa.float64()` is what they already compiled to.

Declared rather than inferred, for the supplied pair: `takes` names the fields
of the struct the *author* packs, the fit half packs from `__FIT__` and the
transform half from `__THIS__`, and only the first is bound when `register` runs.
Inferring would derive from one side and hope the other agreed, against a
docstring that says the declaration is authoritative so a disagreeing transform
refuses rather than mislabelling lanes. A projection is the exception because it
can read both sides out of its own text.

## `Program`: the shared thing

`SQLProjection` does **not** subclass `SQLTransform`. `SQLTransform` is two
things wearing one name — the compiled two-parameter text, and an sklearn
estimator — and a projection wants the first and none of the second.

The compiled text is extracted as a value both classes *hold*
(TASK-87, lands first, behaviour-neutral):

```python
@dataclass(frozen=True, slots=True)
class Program:
    node: Node                       # resolved text, both parameters live
    steps: list[tuple[str, Node]]    # the fit DAG, in dependency order
    residual: Node
    shadowable: set[str]
    bindings: Bindings
    foreign: Foreign
    captured: Captured
    source: str
    connection: Connection | None

    @classmethod
    def compile(cls, sql, scope, *, connection=None, captured=None) -> Self
    def fit(self, data) -> Fitted
    def run(self, data) -> pa.Table   # both parameters bound to one relation
```

`scope` is a parameter, not a `sys._getframe` call inside `Program`: each public
class reads its own caller and passes the mapping in. Moving the frame read one
level deeper would silently break every `FROM df` replacement-scan idiom.

## Binding

Names in a transform are **lexically bound**, and D2 makes that load-bearing
rather than incidental: splicing is substitution, and substitution that
captures is wrong.

The discipline, all of it already implemented in `_transform.py`:

| | |
|---|---|
| free names | resolved from `sys._getframe(1)` at construction and captured **by value**, so rebinding afterwards cannot change what was built |
| CTE names | shadow, case-folded, because DuckDB's binder is case-insensitive and comparing exact strings refused valid SQL |
| recursive CTEs | in scope inside their own body; plain ones are not |
| the `__` prefix | reserved by `_reserve`, so nothing an author writes can collide with anything the model synthesizes |
| `__FIT__`, `__THIS__` | the two parameters — the only names bound at fit/call rather than at construction, and the only `__` names an author may write |
| splicing | `_splice` alpha-renames a member's free names to `{name}__{free}` and re-captures them, and does the same for foreign stems |

A projection carries the same rules, and its leaf role adds nothing new: a
spliced projection's free names are alpha-renamed into its host exactly as a
member transform's are, so the host's frame cannot capture them. That renaming
is also what D3 stands on — the prefix that prevents capture is the prefix that
attributes the error.

One consequence worth stating plainly, because it is the whole reason the
parameters are spelled in capitals: `__FIT__` and `__THIS__` are *not* free
variables with a convention attached. They are the two binders, and every other
name in the text is lexical.

## Two classes, one name

`sql_transform.SQLProjection` (marginalizing) stays where it is.
`sql_transform.model.SQLProjection` (explicit `__FIT__`) is the new one. The
old class still has the transformer registry, per-group fitting, named outputs,
struct outputs and `unnest` — none of which the new one has on day one.

This is real debt, recorded rather than pretended away, and it resolves in two
moves rather than one. The old *class* goes when the new one covers the
transformer registry. `_marginalize.py` does not go with it: it becomes the
guts of `SQLProjection.marginalize`, demoted from the way projections are
authored to a helper that writes the `__FIT__` half for you.

## Properties and gates

| | |
|---|---|
| **rows** | `p.transform(X).num_rows == X.num_rows`, for every fitted `p` and every `X` |
| **solo** | `p.transform(X) == concat(p.transform(X[i:i+1]) for i in range(len(X)))`, as an *ordered* list — strict 1-1 plus threaded row order makes the multiset weakening unnecessary |
| **parity** | `p.transform(X) == pa.Table.from_pylist(p.compile().infer_rows(X))` |
| **faithful** | `p.fit(D).transform(D) == run(p, D)` — inherited from the transform model, not restated |
| **leaf** | a projection spliced into a host produces the same values as the same projection standalone |
| **capture** | a projection whose free name collides with a host name resolves to its own — the alpha-renaming works |
| **attribution** | a spliced projection with a deliberate error produces a message naming the projection (D3) |
| **refusals** | every `NotRowWise` shape in the table above has a test; the gate walks the table looking for gaps, the way `_correlate`'s does for `REASONS` |

`solo` is the one that catches a wrong premise rather than a wrong
implementation. Hand-picked rows are not enough: `avg([1,5,3]) = 3` is a fixed
point, and a probe that lands on one reports success for a batch-dependent
query. The check is the whole batch against the concatenation of every row
served alone, on generated data.

## Slices

All six implemented 2026-08-12, as a stacked PR chain (AmirHossein's call for
this loop; the no-stacking rule stands elsewhere):

1. **TASK-87** — extract `Program`. Behaviour-neutral; zero test edits. **#114**
2. `SQLProjection` + `NotRowWise` + the batch `transform` with threaded row
   order. Gates: `rows`, `solo`, `faithful`, the refusal table. **#115**
3. `KeyNotUnique` at fit, with the cross-join one-row case. **#117**
4. `compile()` → `DuckDBInferFn`. Gate: `parity`. **#118**
5. Splicing a projection as a leaf, θ as data (D1, D2). Gates: `leaf`,
   `capture`. **#119**
6. Attribution (D3), shrunk to its surviving mechanism — see D3's *how it
   landed*. Gate: `attribution`. **#120**

Implementation corrections folded back into this spec: the spine `WHERE` rule
(this file's own draft claim was wrong), the join rows of the refusal table,
`KeyNotUnique`'s zero-row and tiebreak cases, the row path's respelling for
Confit (passthrough inlining; cross join → `LEFT JOIN ON 1 = 1`, since `ON
TRUE` prints back as `CAST('t' AS BOOLEAN)`, which Confit refuses; the row
model trimmed to columns the residual reads), and D3's shrinkage. Two
mechanical traps for the record: `case BaseTable(table_name=THIS)` *binds*
`THIS` instead of comparing it, and struct_pack field aliases ride into a
splice as named arguments (`avg(price := price)`) unless unaliased.

`Transform.takes`/`returns` becoming `pa.Schema` is **not** in this list any
more. D2 removed its projection-side justification: a spliced leaf declares no
struct types. It is still worth doing for supplied Python pairs — an
`OrdinalEncoder` leaf dies at bind today — but it is an independent fix rather
than a prerequisite, and nothing here waits on it.

## Deferred

- **`SQLProjection.marginalize`** — the opt-in helper that derives the `__FIT__`
  half from a `__THIS__`-only text, built on `_marginalize.py`. A rewrite in
  front of the constructor, so it inherits every guarantee below it.
- **The port** — transformer registry, per-group fitting, named outputs, struct
  outputs, `unnest` — which is the deletion trigger for the old *class*.
- **Types beyond the leaf struct.** Widening `pa.Schema` support is per-field
  and complete for the leaf boundary; nothing here turns on the rest of arrow's
  type vocabulary.
