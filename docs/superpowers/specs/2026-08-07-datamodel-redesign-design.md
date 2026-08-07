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

`__FIT__` may appear only in an **uncorrelated** relation.

```sql
SELECT (SELECT avg(price) FROM __FIT__ f WHERE f.cat = t.cat) AS m FROM __THIS__ t
-- CorrelatedFit: __FIT__ subquery references t.cat from the outer query
```

That subquery is per-serving-row, so it cannot be evaluated once into a
table. Freezing it would mean inferring the key `cat`, lifting it to a
`GROUP BY`, and rewriting the correlation into a join — which is
marginalization, arriving through the front door. Refusing keeps freezing
equal to "evaluate once".

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

## Slices

1. **`SQLTransform`, `__FIT__`/`__THIS__`, `fit`/`transform`.** Single level,
   no nesting. Find maximal `__FIT__`-only subtrees, refuse correlated ones,
   freeze at fit, substitute at transform. Gate: the law, plus `CorrelatedFit`.
2. **Nesting and name resolution.** Frame lookup at construction, splice,
   name-prefix, depth cap. Gate: the law on a nested transform, plus spliced
   text equals a hand-written reference.
3. **sklearn leaves.** `x_fit` UDAF / `x_transform` UDF, θ handles, the
   instance registry. Gate: the law with a `StandardScaler` leaf, plus P14 on
   an unseen key.

TDD, red before green. Lands alongside the existing modules in
`sql_transform/model/`; nothing is deleted until the old implementation and
the new one agree on the corpus.

## Deferred

Designed and then cut — reasoning is in commits `6a3b23a` and `c83d89b`, not
lost:

- **`SQLProjection`** as a second type, defined by row-locality of the
  residual. Needs marginalization to define, and `compile()` to be worth
  having.
- **Marginalization** — automatic lifting of windows over `__THIS__` into
  frozen tables. Everything is hand-written here.
- **`compile()` / `Inference`** — the serving artifact and the row path.
- **Freezing a constant.** `memorize(geo)` could freeze a relation over a
  static table that reads neither implicit table. `__FIT__` has no way to say
  it. Pure performance, no semantics; add a marker back if it ever measures.
- **Scalar-position freezing** — `price / avg(price) OVER (...)` reading
  `__FIT__` inline. The ergonomic spelling, and the place key inference
  returns.
