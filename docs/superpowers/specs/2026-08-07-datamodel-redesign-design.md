# Data model redesign — two types, one readable layer (v0)

Date: 2026-08-07. Status: designed with AmirHossein (this session).
Replaces the fit/transform data model, not the surface laws. Every DuckDB
behavior cited was measured on 2026-08-04/06, not recalled.

## What broke

The current model calls something a projection when it maps one input row
to one output row. That test admits window functions, and window functions
are where it fails. Measured, on a projection fitted with
`avg(price) OVER (ORDER BY ts ROWS 2 PRECEDING)`:

```
seen   ts=3, price=999  ->  ma = 20.0    the row's own value is ignored
unseen ts=9, price=50   ->  ma = None
```

Both answers are wrong and neither raises. Compare a partition-keyed window
on the same machinery, which is *right*:

```
price=10 -> 0.5 ;  batch of three 10s -> [0.5, 0.5, 0.5]     mean frozen at fit
```

Same SQL shape, same code path, opposite correctness. The difference is not
cardinality — both are one-row-in, one-row-out. The difference is whether
the output of row *i* depends on rows other than *i*.

Partition keys recur in production, so freezing a partition-keyed window
gives you the learned parameter you wanted. Order keys never recur, so
freezing an ordered frame memorizes the training set and then misses
forever. Silently. That is C5's one unrecoverable state ("no third mode":
every query either serves or refuses by name; a query that builds and
quietly computes something else is unrecoverable).

## The two types

```
SQLTransform     table function.   N -> M.   fit / transform
SQLProjection    UDAF + UDF pair.  N -> N.   fit / transform / compile
```

**The two are independent classes. `SQLProjection` does not inherit from
`SQLTransform`.** The conceptual relation is narrowing — a projection is a
transform that satisfies row-locality — but expressing it as inheritance
would make `isinstance(proj, SQLTransform)` true, and the two types nest
*differently* (table function versus UDAF/UDF pair), so telling them apart is
the one job the types have. An inherited `isinstance` makes every nesting
check order-dependent, which is the same failure already present in the tree
as the `hasattr(obj, "fit") and hasattr(obj, "transform")` duck-check that an
`SQLProjection` passes. The relation is a constructor, not a base class:

```python
class SQLProjection:                            # not a subclass
    def __init__(self, sql, **kw):
        t = SQLTransform(sql, **kw)
        require_row_local(marginalize(t))       # NotRowLocal, at construction
    def as_transform(self) -> SQLTransform: ... # explicit widening
```

Shared implementation lives in functions. Two classes do not earn a
hierarchy.

Two terms used throughout:

- **residual** — what is left of a marginalized transform once the
  `memorize` sections are lifted out into their own relations. It is the
  query that runs at serving time.
- **`run(t, D)`** — execute `t`'s SQL directly against `D` with `__THIS__`
  bound, no fitting and no freezing. The reference behavior.

They are DuckDB's function kinds, which is why composition needs no separate
design: a projection is usable as a scalar UDF *because* it is row-local, a
transform is usable as a table function *because* it is a relation.

`SQLAggregation` was considered and dropped. A keyed reduction is an
`SQLTransform` that returns fewer rows than it took; it needs no type of its
own, and giving it a `fit` would have been a lie — it has nothing to learn.

The defining law:

> **An `SQLProjection` is an `SQLTransform` whose marginalized residual is
> row-local.**

Checked at construction (P7 — "refusals are construction-time and named":
everything refused is refused at construction with an error naming the
construct, never at fit, never at serve, never silently).

```python
SQLProjection("SELECT (price - avg(price) OVER (PARTITION BY cat)) AS z FROM __THIS__")
# marginalizes to a frozen table + a row-local residual  -> admitted

SQLProjection("SELECT avg(price) OVER (ORDER BY ts ROWS 2 PRECEDING) AS ma FROM __THIS__")
# nothing liftable; the residual still holds a window   -> refused at construction
# NotRowLocal: 'ma' is an ordered frame over __THIS__; it cannot be a projection
```

The failure that motivated this redesign is now a type error at
construction, produced by the definition rather than by a special case.

## Row-locality

Row-local means: **the output of row *i* is a function of row *i* and the
fitted state.** The fitted state is constant, so joining into it is
row-local. Joins are not restricted; what you join *to* is.

```sql
-- yes: a frozen table. this is what serving compiles to.
LEFT JOIN memorize(cat_stats) c USING (cat)

-- yes: a static reference table registered with the transform.
LEFT JOIN category_meta m ON t.cat = m.cat

-- no: the right side reads __THIS__, so row i now depends on other rows.
LEFT JOIN (SELECT cat, avg(price) AS m FROM __THIS__ GROUP BY cat) s USING (cat)
```

Three conditions, all mechanically checkable:

| condition | check | why |
| --- | --- | --- |
| right side is constant | not `__THIS__`, not a subquery over it | otherwise the answer depends on the batch |
| `LEFT` join | reject `INNER`/`RIGHT`/`FULL` | anything that can drop a row breaks row-for-row |
| at most one match | `count(*) = count(DISTINCT key)`, once at fit | fan-out breaks row-for-row |

The uniqueness check is one query at fit time, so no key declaration is
required of the user. Tables that `fit` itself produced skip it — `GROUP BY`
already guarantees uniqueness.

Consequences, both deliberate:

- **A registered static table becomes part of the fitted artifact.** If
  serving joins `category_meta`, that table ships with the model. Otherwise
  the same input yields different answers on different machines.
- **Unseen key gives NULL, not a dropped row.** This is P14 ("the one NULL
  story": unseen group ⇒ LEFT JOIN miss ⇒ NULL output; NULL-ness always
  flows through join data, never through a lookup convention), and
  `LEFT`-only is what makes it true.

## `memorize` — the middle layer

`memorize(X)` marks a relation whose value is computed once at fit and
frozen. It is the *only* freezing mechanism in the model.

```sql
WITH cat_stats AS (
    SELECT cat, avg(price) AS a, stddev_pop(price) AS s FROM __THIS__ GROUP BY cat
)
SELECT (t.price - m.a) / m.s AS z
FROM __THIS__ t LEFT JOIN memorize(cat_stats) m USING (cat)
```

This makes `fit` explicable in one sentence:

> **`fit` evaluates every `memorize(...)` once and replaces it with its
> result.**

Three properties follow, and they are the reason for the layer:

**The frozen table takes its name from the CTE.** `ft.tables["cat_stats"]`,
not `__params_0__`. The artifact is inspectable without a decoder ring.

**The user writes the join.** `USING (cat)` is the answer to "how does this
table join back", so no key inference is needed, and no invariant about
inferred keys has to be maintained.

**Deleting `memorize` yields the reference implementation.** The unfrozen
query is the same text minus a function call, which gives every layer a
free oracle (below).

### Admissibility

```
memorize(X) is legal iff X's value does not depend on the serving batch:
  - X is constant (never reads __THIS__), or
  - X is a keyed reduction over __THIS__ (GROUP BY, joined on its keys)
Anything else refuses by name.
```

```python
memorize(SELECT ts, avg(price) OVER (ORDER BY ts ROWS 2 PRECEDING) AS ma FROM __THIS__)
# CannotMemorize: not a keyed reduction — 'ma' would be looked up by ts, which never recurs
```

`memorize` is a request to materialize, never a licence to change meaning.
It is user-writable as well as compiler-emitted: the same syntax, the same
law. That makes it the escape hatch and the thing you hand-edit when
debugging.

The constant case is the one `marginalize` would not find on its own, since
there is nothing to learn — freezing it is pure performance:

```sql
WITH geo AS (SELECT region, count(*) AS n FROM store_dim GROUP BY region)   -- no __THIS__
SELECT t.*, g.n FROM __THIS__ t LEFT JOIN memorize(geo) g USING (region)
```

Freezing `geo` pins `store_dim` as of fit time; later edits to that table do
not reach the fitted model. That is the right default — reproducibility —
but it is a behavior, not a detail.

## The layers, and the gate on each

```
SQLTransform          what you write — arbitrary, windows and all
   |  marginalize     syntax -> syntax
SQLTransform          same type, now explicit about what is frozen
   |  fit(data)       evaluate the memorize sections
FittedTransform       residual + tables
   |  transform(data)
```

`marginalize` is `SQLTransform -> SQLTransform`. That closure is what makes
this a layer rather than a pass into a private IR: the output is a legal
transform you can print, hand-edit, and feed back in.

```python
L1  run(strip_memorize(marginalize(t)), D) == run(t, D)          for all D
L2  t.fit(D).transform(D)                  == run(t, D)
L3  proj.fit(D).compile()(row)             == proj.fit(D).transform(one_row_table)
```

L1 is "marginalize preserves meaning": the rewrite is semantics-preserving
on *any* data, before freezing enters the picture. Fuzzable over the corpus.

L2 is the training-set roundtrip invariant: freezing on the data you fitted
on changes nothing. It holds only at `D`, which is exactly what makes it a
test of *what* got frozen — a transform that froze an ordered frame fails it
the moment `D` has a repeated order key.

L3 is C3 ("binding parity": the row path equals the batch path
value-for-value on the same fitted artifact).

The two must be stated separately. L1 with `t.fit(D)` on the right would be
vacuous, since `fit` marginalizes internally and `marginalize` is
idempotent; `run` is what makes both laws bite.

The point of separating them is that a failure now names a layer. Today a
wrong number tells you the pipeline is broken; here L1 failing means the
rewrite is wrong, L2 failing means the wrong thing was frozen, L3 failing
means the row binding diverged.

## `fit`, `transform`, `compile`

```python
class SQLTransform:
    def fit(self, data) -> "FittedTransform": ...

class SQLProjection:
    def fit(self, data) -> "FittedProjection": ...

@dataclass(frozen=True)
class FittedTransform:
    residual: SQLTransform
    tables:   dict[str, pa.Table]
    def transform(self, data) -> pa.Table: ...

@dataclass(frozen=True)
class FittedProjection:                 # not a subclass of FittedTransform
    residual: SQLProjection
    tables:   dict[str, pa.Table]
    def transform(self, data) -> pa.Table: ...
    def compile(self) -> "Inference": ...
```

`fit` is small enough to read:

```python
def fit(self, data):
    mt = marginalize(self)
    return FittedTransform(strip_memorize(mt), {n: run(x, data) for n, x in memorized(mt)})
```

`compile()` exists only on `FittedProjection`, because L3 holds only when the
residual is row-local. The type system carries the guarantee.

```python
inf = proj.fit(train).compile()
inf(row)           # hot path: struct in, struct out
inf.batch(table)   # same answer, vectorized
inf.schema         # in/out types — for validation, and for codegen later
```

`Inference` is a separate object rather than a method for two reasons: it is
where the one-time cost is paid (params baked into arrays, plan compiled,
boundary warmed — the project's actual cost centre), and it is the
shippable. `FittedProjection` is the training artifact; `Inference` is what
goes to the serving box.

Naming is unsettled — `compile()`/`Inference` is the current choice, with
ONNX's `InferenceSession` as precedent. See open questions.

### `fit` returns a new object

sklearn's `fit` mutates and returns `self`; there is no separate fitted type.
Ours returns a new immutable one. The deviation is deliberate — immutability
is what makes the fitted thing shippable — and it is named rather than
papered over. `SQLTransform` is therefore not an sklearn transformer (no
`transform` before fit) and `FittedTransform` is not an sklearn estimator (no
`fit`). A `Pipeline`-compatible adapter holding both is a thin wrapper, out
of scope here.

## Nesting

```sql
-- transform in transform        FROM other_transform(__THIS__)
-- projection in transform       SELECT p_transform(p_fit(b) OVER (...), b)
-- projection in projection      the same, one level down
-- transform in projection       FROM memorize(other_transform(__THIS__))   <- only frozen
```

A table function is not row-local, so it reaches inside a projection only
through `memorize`; once frozen it is a constant table, and a keyed lookup
into a constant is row-local. The lattice closes on row-locality alone —
there is no separate nesting rule.

Composition (TASK-65) becomes a rewrite you can read. A caller wrapping a
member in `OVER (PARTITION BY store)` merges its key into the member's
frozen sections:

```sql
-- member alone
WITH __m_cat AS (SELECT cat, avg(price) AS a, stddev_pop(price) AS s
                 FROM __THIS__ GROUP BY cat)
SELECT (t.price - m.a) / m.s AS z
FROM __THIS__ t LEFT JOIN memorize(__m_cat) m USING (cat)

-- member under a caller partitioned by store
WITH __m_cat AS (SELECT store, cat, avg(price) AS a, stddev_pop(price) AS s
                 FROM __THIS__ GROUP BY store, cat)
SELECT (t.price - m.a) / m.s AS z
FROM __THIS__ t LEFT JOIN memorize(__m_cat) m USING (store, cat)
```

One column into `GROUP BY`, one into `USING`. That diff previously lived as
`keys ('__cf_k0',) -> ('__cf_k0','__cf_k1')` inside a plan object with no
printable form. TASK-65's acceptance criterion "gate against a hand-written
per-group reference" becomes a text comparison rather than a numeric one,
which is the difference between a test that reports *that* it broke and one
that reports *where*.

## sklearn leaves

An sklearn transformer is a UDAF plus a UDF, so it is a non-SQL
implementation of the two halves the model already has. `marginalize` never
asks which.

```sql
memorize(SELECT cat, sc_fit(struct_pack(v := price)) AS theta FROM __THIS__ GROUP BY cat)
```

There is exactly one freezing mechanism, and SQL and Python leaves are both
expressed in it.

This also collapses the symbolic-inlining work that was previously planned.
There are two member kinds, not three:

```python
transformers={"z": SQLProjection(...)}   # splice the SQL — syntax, not symbolic execution
transformers={"z": StandardScaler()}     # call the UDAF/UDF pair
```

Nothing needs symbolic inlining, because anything inlinable was SQL to begin
with. The inlining gate survives and is cheap: run the spliced form against
the same member called through the UDF boundary, assert equal.

Cost of this choice: a foreign leaf's frozen table holds an opaque handle
rather than columns.

```
keys..., theta          Struct<type,id> into a registry of fitted objects
keys..., mean, scale    what an SQL leaf gives you
```

So "the fitted artifact is just tables" holds for SQL leaves and not for a
fitted `RandomForest`. Accepted.

## Refusals

All at construction (P7), all named. The set is small because the model
derives them rather than enumerating them.

| refusal | condition |
| --- | --- |
| `NotRowLocal` | `SQLProjection(...)` whose marginalized residual reads other rows |
| `CannotMemorize` | `memorize(X)` where X is neither constant nor a keyed reduction |
| `UnsafeJoin` | non-`LEFT` join, or a join to a relation reading `__THIS__`, inside a projection |
| `AmbiguousJoin` | frozen table's join key is not unique (checked once at fit) |

An unmemorized window inside an `SQLTransform` is **not** a refusal. It means
what DuckDB means — a window over the rows you handed it. That is honest and
oracle-faithful (C2 — "engine parity": bit-for-bit identical to DuckDB with
the same UDFs registered, or a named refusal, no third behavior). It simply
makes the fitted transform batch-dependent, which is why it cannot be an
`SQLProjection` and cannot be compiled.

## What this deletes

`_marginalize.py` is 2641 lines. The following are removed outright rather
than ported:

- window admissibility analysis — which window shapes may be lifted
- `DISTINCT`-based params extraction (also ~3x slower than `GROUP BY`,
  measured across three shapes)
- join-key inference and the invariant that every admitted window's value is
  a function of its inferred keys — the user writes the join
- key merging as an opaque plan mutation — it is now visible SQL
- the separate `SQLAggregation` type
- symbolic inlining of foreign estimators

## Build plan

New model lands **alongside** the existing one, which serves as its oracle
until the corpus agrees; nothing is deleted before then. Cost is a
temporarily duplicated tree.

New subpackage `sql_transform/model/`. Existing flat modules untouched.

1. **Types and row-locality.** `SQLTransform`, `SQLProjection`, the
   row-locality checker, the three join conditions. No `marginalize` yet —
   only hand-written SQL. Gate: construction refuses correctly, including
   the ordered-frame case.
2. **`memorize`.** Parse, freeze, join back. Hand-written `memorize` only.
   `fit`/`transform`. Gate: L2 on hand-written transforms.
3. **`marginalize`.** Windows to `memorize`. Gate: L1 fuzzed over the corpus,
   plus L2.
4. **`compile()` and `Inference`.** Gate: L3.
5. **sklearn leaves.** UDAF/UDF pair, opaque handles. Gate: existing
   transformer tests ported.
6. **Nesting and key merge.** TASK-65 lands here. Gate: marginalized text
   equals a hand-written reference.
7. **Differential corpus** old vs new. When green, delete the old modules and
   flatten `model/` up.

Each slice is TDD, red before green.

## Open questions

- **Scalar-position `memorize`.** `price / memorize(avg(price) OVER (PARTITION BY cat))`
  is the spelling people will write, and ergonomic authoring is a stated
  project goal. Definable as sugar for the table form — lift to a frozen
  table keyed by the partition keys, join back — but it is the one place key
  inference returns. Table form ships first; decide before slice 3.
- **Does `memorize(cat_stats)` survive DuckDB's parser?** We rewrite before
  execution so it never has to be a registered function, but it must parse.
  Cheap to check; blocks slice 2.
- **Naming.** `compile()`/`Inference` versus `infer()`, and whether
  `marginalize` keeps its name now that its output is readable.
- **`Pipeline` adapter.** A wrapper presenting the sklearn estimator
  protocol over the `SQLTransform`/`FittedTransform` pair. Wanted eventually,
  not scoped here.
- **Cost of the uniqueness check** on large registered static tables. One
  `count(DISTINCT)` at fit; measure before assuming it is free.
