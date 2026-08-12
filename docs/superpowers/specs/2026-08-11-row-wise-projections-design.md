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

This mirrors `docs/superpowers/specs/2026-08-05-fit-transform-split-design.md`,
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

| Shape | Why |
|---|---|
| aggregate over a `__THIS__` column | the answer depends on the other rows in the batch |
| window function over `__THIS__` | same |
| `GROUP BY` / `HAVING` | same |
| `DISTINCT` | drops rows, and which rows depends on the batch |
| `ORDER BY` / `LIMIT` / `OFFSET` | destroys the row correspondence |
| `__THIS__` referenced more than once | a self-join is cross-row |
| a set operation with `__THIS__` in an arm | cross-row |
| `WHERE` / `QUALIFY` reading a `__THIS__` column | drops rows, and a scalar UDF has no encoding for *no row here* |
| a recursive CTE reading `__THIS__` | cross-row by construction |

A `WHERE` that reads only params columns is fine — it is constant at serving.

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
and it is row-wise precisely because the subquery returns one row.

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

`transform` is the oracle — DuckDB, batch. `infer`/`infer_batch` go through
Confit's `DuckDBInferFn` with `shape="map"`, the same residual and the same
params tables. Confit's contract makes the two bit-exact or refuses by name.

`shape="map"` is forced, not chosen: it is the same fact as `tfm_transform`
being a scalar UDF, seen from the serving side. `infer` returns a row, never
`None`, and no caller downstream gains an optional case.

```python
p = SQLProjection(sql)
fitted = p.fit(TRAIN)

fitted.transform(X)              # pa.Table, num_rows == X.num_rows
fitted.infer({"age": 31})        # one typed row, never None
fitted.infer_batch(rows)         # list of typed rows
```

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

```python
fitted.as_leaf() -> Transform
```

`_foreign.Transform` already is the pair; a projection fills it in:

- `fit(F)` runs the freeze steps against `F` and returns the params as the
  instance — inspectable and shippable, which is what `_foreign.py`'s docstring
  already promises an SQL leaf would give.
- `transform(instance, T)` runs the residual with those params registered.
- `takes` / `returns` are **computed from the projection's own text**, not
  declared.

Used under a window, `p_fit(…) OVER (PARTITION BY country)` fits one projection
per country — `__FIT__` binds to the group's rows. That falls out of `x_fit`
being a UDAF; nothing extra is needed for it.

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
| **parity** | `p.transform(X) == pa.Table.from_pylist(p.infer_batch(X))` |
| **faithful** | `p.fit(D).transform(D) == run(p, D)` — inherited from the transform model, not restated |
| **leaf** | a projection used as a leaf produces the same values as the same projection standalone |
| **refusals** | every `NotRowWise` shape in the table above has a test; the gate walks the table looking for gaps, the way `_correlate`'s does for `REASONS` |

`solo` is the one that catches a wrong premise rather than a wrong
implementation. Hand-picked rows are not enough: `avg([1,5,3]) = 3` is a fixed
point, and a probe that lands on one reports success for a batch-dependent
query. The check is the whole batch against the concatenation of every row
served alone, on generated data.

## Slices

1. **TASK-87** — extract `Program`. Behaviour-neutral; if a test needs editing
   beyond an import line, the diff is wrong.
2. `Transform.takes`/`returns` become `pa.Schema`. Eight edits, no behaviour
   change.
3. `SQLProjection` + `NotRowWise` + the batch `transform` with threaded row
   order. Gates: `rows`, `solo`, `faithful`, the refusal table.
4. `KeyNotUnique` at fit, with the cross-join one-row case.
5. `infer` / `infer_batch` through Confit. Gate: `parity`.
6. `as_leaf()`. Gate: `leaf`.

Each lands as its own PR off master, sequentially.

## Deferred

- **`SQLProjection.marginalize`** — the opt-in helper that derives the `__FIT__`
  half from a `__THIS__`-only text, built on `_marginalize.py`. A rewrite in
  front of the constructor, so it inherits every guarantee below it.
- **The port** — transformer registry, per-group fitting, named outputs, struct
  outputs, `unnest` — which is the deletion trigger for the old *class*.
- **Types beyond the leaf struct.** Widening `pa.Schema` support is per-field
  and complete for the leaf boundary; nothing here turns on the rest of arrow's
  type vocabulary.
