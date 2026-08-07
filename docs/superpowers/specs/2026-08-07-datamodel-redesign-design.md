# SQLTransform, `__FIT__`, nesting (v0)

Date: 2026-08-07. Status: designed with AmirHossein (this session).
Scope deliberately small: one type, two implicit tables, one nesting rule.
Marginalization and the serving path are deferred — see the end.

## The type

```python
class SQLTransform:
    def __init__(self, sql): ...          # references resolve from the caller's frame
    def fit(self, data) -> "FittedTransform": ...

@dataclass(frozen=True)
class FittedTransform:
    residual:  SQLTransform
    frozen:    dict[str, pa.Table]
    instances: dict[str, Any]             # fitted estimators, see sklearn below
    def transform(self, data) -> pa.Table: ...
```

A table function. `N -> M`, arbitrary SQL. No shape constraint, no second
type.

## `__FIT__` and `__THIS__`

Two implicit tables:

- **`__FIT__`** — the training data. Read at fit, frozen, never read again.
- **`__THIS__`** — the batch being transformed. Read at transform.

```sql
WITH cs AS (SELECT cat, avg(price) AS m FROM __FIT__ GROUP BY cat)
SELECT t.price / f.m AS rel FROM __THIS__ t LEFT JOIN cs f USING (cat)
```

Which half is learned and which is live is *read off the text*. There is no
annotation to remember and none to forget.

### The freezing rule

> **Every maximal subquery whose leaves are all `__FIT__` and constants is
> evaluated once at fit and replaced by a table.**

"Maximal" does the work. A relation mixing both tables is not frozen
wholesale — its `__FIT__`-only subtrees are, and the rest stays live:

```sql
SELECT * FROM __THIS__ t JOIN (SELECT cat, avg(price) m FROM __FIT__ GROUP BY cat) f
       USING (cat)                    -- the subquery freezes; the join runs per batch
```

`fit` is one sentence, and the implementation is a base-table name lookup —
no marker syntax, no function node, nothing to parse specially:

```python
def fit(self, data):
    frozen = {name: run(sub, data) for name, sub in fit_only_subtrees(self)}
    return FittedTransform(substitute(self, frozen), frozen, instances)
```

A transform that never mentions `__FIT__` is stateless: `fit` is a no-op and
`transform` is the query.

### The one refusal

A `__FIT__` subtree may not correlate to **`__THIS__`**.

Correlation *inside* a closed `__FIT__` subtree is fine — it is evaluated once
as a whole, so nothing about it is per-serving-row. That distinction is what
makes per-group members work (below) without new machinery.

```sql
SELECT (SELECT avg(price) FROM __FIT__ f WHERE f.cat = t.cat) AS m FROM __THIS__ t
-- CorrelatedFit: __FIT__ subquery references t.cat from the outer query
```

That subquery is per-serving-row, so it cannot be evaluated once into a
table. Supporting it means inferring the key `cat`, lifting it to a
`GROUP BY`, and rewriting the correlation into a join — which is exactly
marginalization. **Future work, not a permanent boundary**: it is the natural
first surface for marginalization to enable, and it is deferred with it. Here
the refusal keeps freezing equal to "evaluate once".

Refused at construction (P7 — "refusals are construction-time and named":
everything refused is refused at construction with an error naming the
construct, never at fit, never at serve, never silently).

Nothing else refuses.

### The ordered frame is legal here

This was the failure that started the redesign. Measured on the current
implementation, which freezes it silently:

```
seen   ts=3, price=999  ->  ma = 20.0    the row's own value is ignored
unseen ts=9, price=50   ->  ma = None
```

Under `__FIT__` the same computation is legal, and written out:

```sql
WITH ma AS (SELECT ts, avg(price) OVER (ORDER BY ts ROWS 2 PRECEDING) AS m FROM __FIT__)
SELECT t.price, ma.m FROM __THIS__ t LEFT JOIN ma USING (ts)
```

It says *join the training data on `ts`*. The hazard has not gone away — a
serving row with a new `ts` still gets NULL forever — but it is now visible
in the text instead of hidden behind an inferred key, and the NULL arrives
through P14 ("the one NULL story": unseen group ⇒ LEFT JOIN miss ⇒ NULL
output; NULL-ness always flows through join data, never through a lookup
convention).

**Deliberately not refused.** The refusal existed because freezing was
implicit; once the author writes the join key themselves, refusing means
overruling something they can see. This is the one place the design is more
permissive than its predecessor, and the trade is explicitness for
permissiveness.

## Name resolution

Unresolved identifiers resolve against the caller's frame, exactly as
`duckdb.sql("SELECT * FROM my_df")` does. Locals first, then globals.

```python
store_dim = pa.table(...)                          # a static table
z         = SQLTransform("... FROM __FIT__ ...")   # another transform

outer = SQLTransform("SELECT * FROM z(__THIS__) WHERE z > 0")
```

No `transforms=` or `tables=` kwargs. What a name resolves to determines how
it is used: an `SQLTransform` is spliced as a table function, an Arrow table
or DataFrame becomes a static table.

**Resolution happens once, at construction, and captures by value.** The
frame is not retained. Rebinding `z` afterwards does not change `outer`, and
nothing holds a reference to a dead stack frame. A name resolving to nothing
refuses at construction, naming the identifier.

Skipped: an explicit-binding escape hatch for transforms built where the name
is not a local (a loop, a factory). DuckDB keeps `con.register` for exactly
this; add it when it bites.

## Nesting

A transform is usable inside another as a table function:

```python
z     = SQLTransform("SELECT t.price / f.m AS z FROM __THIS__ t "
                     "LEFT JOIN (SELECT cat, avg(price) m FROM __FIT__ GROUP BY cat) f USING (cat)")
outer = SQLTransform("SELECT * FROM z(__THIS__) WHERE z > 0")
```

Splice the member's SQL at the call site, prefixing its CTE names with the
binding name to avoid collisions. Both implicit tables pass through
unchanged — the member's `__FIT__` is the outer's `__FIT__` — so one `fit`
freezes everything, and `ft.frozen["z__cs"]` says where each table came from.

Nesting depth is capped at 8, refusing by name beyond that. Same cap the
previous composition design used; no reason found to change it.

### Applying a member per group

A member is `N -> M`, so "fit it per store" cannot be a window — windows do
not change cardinality. The mechanism is `LATERAL`, and it needs nothing new,
because the whole construct is a closed `__FIT__` subtree and freezes as one
unit:

```sql
WITH per_store AS (
    SELECT g.store, x.*
    FROM (SELECT DISTINCT store FROM __FIT__) g,
         LATERAL (SELECT cat, avg(price) AS a FROM __FIT__ WHERE store = g.store GROUP BY cat) x
)
SELECT t.price / p.a AS rel FROM __THIS__ t LEFT JOIN per_store p USING (store, cat)
```

**Splice members, never emit a DuckDB macro for them.** Measured 2026-08-07 on
1.5.5: a table macro invoked under `LATERAL` does not see the correlation and
silently returns the whole-table answer for every group.

```
LATERAL <spliced body>   S1/book=15,   S1/toy=30,   S2/book=100,  S2/toy=250    correct
LATERAL m('t')           S1/book=43.3, S1/toy=176.7, S2/book=43.3, S2/toy=176.7 wrong
```

It runs, returns the right shape, and is wrong — C5's silent-wrongness case.
Splicing avoids it entirely, which is one more reason the member surface is
text substitution rather than catalog objects.

**Key-folding is not a general alternative.** Rewriting the member's
`GROUP BY cat` into `GROUP BY store, cat` — what the previous composition
design did — agrees with `LATERAL` only when the member's relation is a keyed
reduction. Measured on a member that is not one:

```
member = avg(price)       fold key  [('S1',20), ('S2',200)]                    == LATERAL
member = top 2 by price   fold key  [('S2',300), ('S2',200)]                   wrong: LIMIT applies after grouping
                          LATERAL   [('S1',30),('S1',20),('S2',300),('S2',200)] correct
```

The old design was safe only because a fit is always a reduction. `LATERAL`
is correct without that precondition, so it is the mechanism; key-folding is
an optimization to consider later, gated on a reduction check.

## sklearn transformers

An sklearn transformer is a UDAF (`fit`) plus a UDF (`transform`). Its fit
reads `__FIT__`, so it freezes by the same rule as everything else — no
mechanism of its own.

```python
sc = StandardScaler()

t = SQLTransform("""
    WITH params AS (
        SELECT cat, sc_fit(struct_pack(v := price)) AS theta FROM __FIT__ GROUP BY cat
    )
    SELECT sc_transform(f.theta, struct_pack(v := t.price)).v AS z
    FROM __THIS__ t LEFT JOIN params f USING (cat)
""")
```

`sc` resolves from the caller's frame; `sc_fit` and `sc_transform` are found
by stripping the suffix. Global fit is the same shape with no keys:

```sql
WITH params AS (SELECT sc_fit(struct_pack(v := price)) AS theta FROM __FIT__)
SELECT sc_transform(f.theta, struct_pack(v := t.price)).v AS z
FROM __THIS__ t, params f
```

**θ is an opaque handle**, `Struct<type, id>` into a registry of fitted
estimators — which is why `FittedTransform` carries `instances` alongside
`frozen`. "The artifact is just tables" holds for SQL leaves and not for a
fitted `RandomForest`: an SQL-expressible transformer gives you an
inspectable, shippable params table, a foreign one gives you a pointer. An id
*present but missing from `instances`* is a broken artifact and raises.

`sc_fit` over `__THIS__` rather than `__FIT__` is legal, and means what it
says: refit on the batch you were handed, at every call. That is the
transductive case. Not a refusal — and under this design you cannot write it
by accident, because you had to type `__THIS__`.

Skipped: the bare `sc(x)` sugar for global fit-transform. It expands to the
cross-join form above; add it when the spelling annoys someone.

## The gate

One law, and it needs no rewriting to state:

```python
run(t, D)  =  execute(t.sql, __FIT__ := D, __THIS__ := D)
t.fit(D).transform(D) == run(t, D)
```

Freezing on the data you fitted on changes nothing. The reference side is the
same SQL text with two names bound to one table — a **binding**, not an AST
edit. The test oracle is the part least worth being clever about, and here it
is nothing.

The law holds only at `D`, which is what makes it a test of *what* got frozen
rather than a tautology. Property-test it over the corpus.

## Mechanism, measured

Measured 2026-08-07 against DuckDB 1.5.5, on the earlier `memorize` spelling
but the substitution mechanism is identical:

- The repo parses with DuckDB's own `json_serialize_sql` /
  `json_deserialize_sql`, so the AST round-trips cleanly.
- Replacing a relation node with a `BASE_TABLE` node naming a frozen table
  works end to end: the law held on training data, and a new batch
  demonstrably used frozen parameters (`[0.5, 6.66]`) rather than recomputing
  (`[1.0, 1.0]`).

`__FIT__` needs none of the parser work that spelling did. It is an ordinary
identifier, so there is no marker to parse, no function node to match, and no
orphaned CTE to prune after substitution.

## Properties and gates

Every property is named descriptively rather than numbered, so citing one in
a review carries its meaning. Where a property restates an existing law from
`docs/properties.md` or `docs/kpis.md`, that is noted.

### Freezing

| Property | Statement | Gate |
| --- | --- | --- |
| **Freezing is complete** | Every maximal `__FIT__`-only subtree appears in `frozen`; nothing reading `__FIT__` survives into the residual | Construct a transform with two `__FIT__` CTEs and one mixed relation; assert `frozen` has exactly the two, and `"__FIT__" not in residual.sql` |
| **Freezing is faithful** | `t.fit(D).transform(D) == run(t, D)`, where `run` binds `__FIT__` and `__THIS__` both to `D` | Property test over the corpus. The reference side is a binding, not a rewrite |
| **Freezing is observable** | For `D' != D`, `fit(D).transform(D')` uses `D`'s parameters | Measured: frozen `[0.5, 6.66]` vs recomputed `[1.0, 1.0]`. **Load-bearing — see below** |
| **Freezing is deterministic** | `fit(D)` twice yields equal `frozen` tables | Equality on two fits of the same data |
| **Fit ignores `__THIS__`** | `fit` reads only `__FIT__`; no serving batch need exist | `fit(D)` succeeds with nothing bound to `__THIS__` |
| **Statelessness is real** | A transform never naming `__FIT__` has `frozen == {}` and a no-op `fit` | `fit(D1).transform(X) == fit(D2).transform(X)` for arbitrary `D1 != D2` |
| **Substitution is surgical** | The residual differs from the original *only* at the frozen subtrees | Text comparison against a hand-written expected residual |

**"Freezing is observable" is what keeps "freezing is faithful" from being
vacuous.** An implementation that freezes nothing at all — `fit` a no-op,
`transform` re-running the whole query with `__FIT__` bound to the serving
batch — passes the faithfulness law on every input. The law only bites when
paired with a second data set. Any test suite that pins the first without the
second is testing nothing.

### Refusals

All at construction (P7 — "refusals are construction-time and named":
everything refused is refused at construction with an error naming the
construct, never at fit, never at serve, never silently).

| Property | Statement | Gate |
| --- | --- | --- |
| **`__THIS__`-correlated `__FIT__` refuses** | A `__FIT__` subquery referencing a `__THIS__` column raises `CorrelatedFit` | Construction raises; the message names the correlated column. Assert it raises at construction, **not** at fit or transform |
| **Internal correlation is allowed** | A `LATERAL` inside a closed `__FIT__` subtree is admitted and freezes as one table | The per-group example constructs, fits, and matches a hand-written per-group reference |
| **Unknown name refuses** | A name resolving to nothing raises, naming the identifier | Construction with an unbound name; message contains the name |
| **Depth cap holds** | Nesting deeper than 8 refuses by name | 9-deep construction raises; 8-deep succeeds |
| **No third mode** | Every transform either serves or refuses by name (C5 — silent wrongness is the one unrecoverable state) | Corpus FAILED bucket pinned empty |

### Deliberate permissiveness

These are cases the design *allows* and a previous draft refused. Each needs a
gate that pins the current behaviour, so a future change cannot quietly alter
it.

| Property | Statement | Gate |
| --- | --- | --- |
| **Ordered frames over `__FIT__` are legal** | A window with an order-keyed frame over `__FIT__` freezes and joins by the order key | Pin the measured behaviour: a seen key returns the training value *ignoring the row's own*, an unseen key returns NULL. Both asserted explicitly, with a comment saying this is intended |
| **`__THIS__`-side aggregation is live** | An aggregate over `__THIS__` means what DuckDB means — recomputed per batch | `transform(X1)` and `transform(X2)` give batch-dependent answers; nothing appears in `frozen` |
| **Transductive `sc_fit` is legal** | `sc_fit` over `__THIS__` refits per call | Two transforms of different batches yield different parameters |

The ordered-frame gate is the uncomfortable one, and that is why it is here.
It pins a result that *looks like a bug* — hence the requirement that the test
carry a comment stating it is intended, so nobody "fixes" it in a year.

### Name resolution

| Property | Statement | Gate |
| --- | --- | --- |
| **Capture is by value** | Rebinding a name after construction does not change the built transform | Build `outer` from `z`, rebind `z`, assert `outer` unchanged |
| **No frame is retained** | Construction holds no reference to the caller's stack frame | Construct inside a function, drop it, assert via `gc`/`weakref` that the frame is collected |

### Nesting

| Property | Statement | Gate |
| --- | --- | --- |
| **Implicit tables pass through** | A member's `__FIT__` is the outer's `__FIT__`; one `fit` freezes everything | Nested transform's `frozen` contains the member's tables, prefixed |
| **Prefixing avoids collisions** | Two members each with a CTE named `cs` yield `z__cs` and `y__cs` | Both keys present in `frozen`, values distinct |
| **Splicing is text-checkable** | The spliced SQL equals a hand-written equivalent | Text comparison — the gate that reports *where* it broke, not just *that* it did |
| **Members splice, never macro** | No DuckDB macro is created for a member | A member applied per group via `LATERAL` returns per-group values, not the whole-table value repeated. This is the gate that catches a macro-based implementation |

### sklearn leaves

| Property | Statement | Gate |
| --- | --- | --- |
| **Leaves need no special case** | An sklearn leaf freezes by the same rule as SQL | The faithfulness law with a `StandardScaler` leaf |
| **The one NULL story** | Unseen key ⇒ LEFT JOIN miss ⇒ NULL θ ⇒ NULL output, never a dropped row (P14) | Transform a batch with an unseen key; the row is present, the output is NULL |
| **Broken artifacts raise** | A θ id present but absent from `instances` raises, never NULL | Delete an instance, transform, expect a raise naming the id |

## Slices

1. **`SQLTransform`, `__FIT__`/`__THIS__`, `fit`/`transform`.** Single level,
   no nesting. Find maximal `__FIT__`-only subtrees, refuse correlated ones,
   freeze at fit, substitute at transform.
   Gates: all of **Freezing**, `CorrelatedFit` and **No third mode** from
   **Refusals**, and both `__THIS__`-side entries under **Deliberate
   permissiveness**.
2. **Nesting and name resolution.** Frame lookup at construction, splice,
   name-prefix, depth cap.
   Gates: all of **Name resolution** and **Nesting**, plus **Unknown name
   refuses** and **Depth cap holds**.
3. **sklearn leaves.** `x_fit` UDAF / `x_transform` UDF, θ handles, the
   instance registry.
   Gates: all of **sklearn leaves**, plus **Transductive `sc_fit` is legal**.

The ordered-frame gate belongs to slice 1 and should be written first in that
slice, not last. It is the property most likely to be "corrected" by someone
who does not know it is deliberate.

TDD, red before green. Lands alongside the existing modules in
`sql_transform/model/`; nothing is deleted until the old implementation and
the new one agree on the corpus.

## Deferred

Designed and then cut — reasoning is in commits `6a3b23a` and `c83d89b`, not
lost:

- **`SQLProjection`** as a second type, defined by row-locality of the
  residual. Needs marginalization to define, and `compile()` to be worth
  having.
- **Correlated `__FIT__` subqueries.** `(SELECT avg(price) FROM __FIT__ f
  WHERE f.cat = t.cat)` — refused by `CorrelatedFit` in slice 1. Enabling it
  is lifting the correlation to a `GROUP BY` and rewriting it as a join, so
  it lands with marginalization and is the first thing marginalization should
  buy.
- **Marginalization** — automatic lifting of windows over `__THIS__` into
  frozen tables. Everything is hand-written here.
- **`compile()` / `Inference`** — the serving artifact and the row path.
- **Freezing a constant.** `memorize(geo)` could freeze a relation over a
  static table that reads neither implicit table. `__FIT__` has no way to say
  it. Pure performance, no semantics; add a marker back if it ever measures.
- **Scalar-position freezing** — `price / avg(price) OVER (...)` reading
  `__FIT__` inline. The ergonomic spelling, and the place key inference
  returns.
