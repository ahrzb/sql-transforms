# SQLTransform, memorize, nesting (v0)

Date: 2026-08-07. Status: designed with AmirHossein (this session).
Scope deliberately small: one type, one freezing mechanism, one nesting rule.
Marginalization and the serving path are deferred — see the end.

## The type

```python
class SQLTransform:
    def __init__(self, sql): ...          # references resolve from the caller's frame
    def fit(self, data) -> "FittedTransform": ...

@dataclass(frozen=True)
class FittedTransform:
    residual: SQLTransform
    frozen:   dict[str, pa.Table]
    def transform(self, data) -> pa.Table: ...
```

A table function. `N -> M`, arbitrary SQL over `__THIS__`. No constraint on
shape, no row-locality requirement, no second type.

## `memorize`

`memorize(X)` marks a relation whose value is computed once at fit and
frozen. It is the only freezing mechanism, and it is hand-written — nothing
infers it.

```sql
WITH cat_stats AS (
    SELECT cat, avg(price) AS a, stddev_pop(price) AS s FROM __THIS__ GROUP BY cat
)
SELECT (t.price - m.a) / m.s AS z
FROM __THIS__ t LEFT JOIN memorize(cat_stats) m USING (cat)
```

`fit` is one sentence:

> **`fit` evaluates every `memorize(...)` once against the training data and
> replaces it with a frozen table.**

```python
def fit(self, data):
    return FittedTransform(strip_memorize(self),
                           {n: run(x, data) for n, x in memorized(self)})
```

Three properties, and they are why this is worth building before anything
else:

**The frozen table keeps the CTE's name.** `ft.frozen["cat_stats"]`, not
`__params_0__`. The artifact is readable without a decoder ring.

**The user writes the join.** `USING (cat)` answers "how does this table join
back", so nothing has to infer keys.

**Deleting `memorize` gives the reference implementation.** The unfrozen
query is the same text minus a function call, which is where the gate comes
from.

### Admissibility

```
memorize(X) is legal iff X's value does not depend on the serving batch:
  - X is constant (never reads __THIS__), or
  - X is a keyed reduction over __THIS__ (GROUP BY, joined on its keys)
Otherwise: CannotMemorize, at construction.
```

```python
memorize(SELECT ts, avg(price) OVER (ORDER BY ts ROWS 2 PRECEDING) AS ma FROM __THIS__)
# CannotMemorize: not a keyed reduction — 'ma' would be looked up by ts, which never recurs
```

That refusal is the whole reason this design exists. Freezing an ordered
frame memorizes the training set: an unseen timestamp returns NULL forever,
and a seen one returns the training answer while ignoring the incoming row's
own value. Measured on the current implementation, which does exactly this
without complaint:

```
seen   ts=3, price=999  ->  ma = 20.0    the row's own value is ignored
unseen ts=9, price=50   ->  ma = None
```

Both wrong, neither raises. That is C5's one unrecoverable state ("no third
mode": every query either serves or refuses by name; a query that builds and
quietly computes something else is unrecoverable). `memorize` makes freezing
explicit and then refuses the case that cannot be frozen honestly.

Nothing else refuses. An unmemorized window is legal and means what DuckDB
means — a window over the rows you handed it. The transform is then
batch-dependent, which is correct, because that is what was written.

The constant case is worth having on its own:

```sql
WITH geo AS (SELECT region, count(*) AS n FROM store_dim GROUP BY region)   -- no __THIS__
SELECT t.*, g.n FROM __THIS__ t LEFT JOIN memorize(geo) g USING (region)
```

Freezing `geo` pins `store_dim` as of fit time; later edits do not reach the
fitted model. Right default — reproducibility — but a behavior, not a detail.

### Spelling, and how the rewrite works

Measured 2026-08-07 against DuckDB 1.5.5. `memorize` never has to be a
registered function: it only has to parse, survive `json_serialize_sql` →
edit → `json_deserialize_sql`, and be findable in the AST. Two spellings
qualify, and they are the same AST node with different child shapes:

```sql
-- named CTE
WITH cs AS (SELECT cat, avg(price) AS m FROM __THIS__ GROUP BY cat)
SELECT t.price / f.m AS rel FROM __THIS__ t LEFT JOIN memorize(cs) f USING (cat)

-- inline, no CTE needed
SELECT * FROM memorize((SELECT cat, avg(price) AS m FROM __THIS__ GROUP BY cat))
```

DuckDB reports a *Catalog* error for these, not a *Parser* error — the text
is valid SQL. `fit` replaces the `TABLE_FUNCTION` node with a `BASE_TABLE`
node naming the frozen table. Measured end to end: the law holds on the
training data, and a new batch demonstrably uses the frozen parameters
(`[0.5, 6.66]`) rather than recomputing (`[1.0, 1.0]`).

Rejected spellings, with the reason each failed:

| spelling | why not |
| --- | --- |
| `WITH cs AS /*+ memorize */ (…)` | DuckDB drops comments from the AST — the marker does not survive serialization |
| `WITH __memorize__cs AS (…)` | survives, but the marker appears at both definition and use, and the convention is invisible at the call site |
| `WITH cs AS MATERIALIZED (…)` | parses *and binds* natively — it already means something to DuckDB, so our reading would silently differ from the oracle's on identical text |
| `memorize(TABLE cs)` | Parser Error |

Two implementation notes from the spike:

- **Prune unreferenced CTEs after the swap.** The memorized CTE remains in
  the serving text once its only reference is replaced. DuckDB will not
  evaluate it, so it is harmless — but leaving the training computation in
  the serving query undercuts the point of a readable layer.
- **The argument arrives as a column reference.** `memorize(cs)` yields
  `function.children[0].column_names[0] == "cs"`, indistinguishable at the
  AST level from a column named `cs`. Unambiguous in `FROM` position, but the
  resolver must know that rather than match on node type alone.

## Name resolution

Unresolved identifiers in the SQL resolve against the caller's frame, exactly
as `duckdb.sql("SELECT * FROM my_df")` does. Locals first, then globals.

```python
store_dim = pa.table(...)                                    # a static table
z         = SQLTransform("... memorize(cat_stats) ...")      # another transform

outer = SQLTransform("SELECT * FROM z(__THIS__) WHERE z > 0")
```

No `transforms=` or `tables=` kwargs. What a name resolves to determines how
it is used: an `SQLTransform` gets spliced as a table function, an Arrow
table or DataFrame becomes a static table.

**Resolution happens once, at construction, and captures by value.** The
frame is not retained. Rebinding `z` afterwards does not change `outer`, and
nothing holds a reference to a dead stack frame. A name that resolves to
nothing refuses at construction, naming the identifier.

Skipped: an explicit-binding escape hatch for transforms built where the
name is not a local (a loop, a factory). DuckDB keeps `con.register` for
exactly this; add it when it bites.

## Nesting

A transform is usable inside another transform as a table function:

```python
z     = SQLTransform("... LEFT JOIN memorize(cat_stats) m USING (cat)")
outer = SQLTransform("SELECT * FROM z(__THIS__) WHERE z > 0")
```

Splice the member's SQL at the call site; prefix its CTE names with the
member's binding name to avoid collisions:

```sql
WITH z__cat_stats AS (SELECT cat, avg(price) AS a, ... FROM __THIS__ GROUP BY cat)
SELECT * FROM (
    SELECT (t.price - m.a) / m.s AS z
    FROM __THIS__ t LEFT JOIN memorize(z__cat_stats) m USING (cat)
) WHERE z > 0
```

The member's `memorize` sections become the outer's, so one `fit` freezes
everything. `ft.frozen["z__cat_stats"]` — still readable, and the prefix says
where it came from.

Nesting depth is capped at 8, refusing by name beyond that. Same cap the
previous composition design used; no reason found to change it.

## sklearn transformers

An sklearn transformer is a UDAF (`fit`) plus a UDF (`transform`), so it
needs no mechanism of its own — its fit is a keyed reduction, which is
exactly what `memorize` freezes.

```python
sc = StandardScaler()

t = SQLTransform("""
    WITH params AS (
        SELECT cat, sc_fit(struct_pack(v := price)) AS theta FROM __THIS__ GROUP BY cat
    )
    SELECT sc_transform(f.theta, struct_pack(v := t.price)).v AS z
    FROM __THIS__ t LEFT JOIN memorize(params) f USING (cat)
""")
```

`sc` resolves from the caller's frame like any other name; `sc_fit` and
`sc_transform` are found by stripping the suffix. Global fit is the same
shape with no keys and a cross join:

```sql
WITH params AS (SELECT sc_fit(struct_pack(v := price)) AS theta FROM __THIS__)
SELECT sc_transform(f.theta, struct_pack(v := t.price)).v AS z
FROM __THIS__ t, memorize(params) f
```

There is one freezing mechanism, and SQL and Python leaves are both
expressed in it.

**θ is an opaque handle**, `Struct<type, id>` into a registry of fitted
estimators. So the frozen table holds ids, not columns, and the artifact
carries both:

```python
@dataclass(frozen=True)
class FittedTransform:
    residual:  SQLTransform
    frozen:    dict[str, pa.Table]   # id per key
    instances: dict[str, Any]        # id -> the fitted estimator
```

"The fitted artifact is just tables" holds for SQL leaves and not for a
fitted `RandomForest`. Accepted — an SQL-expressible transformer gives you an
inspectable, shippable params table; a foreign one gives you a pointer.

An unseen key misses the `LEFT JOIN`, giving a NULL θ and a NULL output —
P14 ("the one NULL story": unseen group ⇒ LEFT JOIN miss ⇒ NULL output;
NULL-ness always flows through join data, never through a lookup
convention). An id *present but missing from `instances`* is a broken
artifact and raises.

**`sc_fit` outside a `memorize` is legal**, and means what DuckDB means: fit
on the batch you were handed, at every call. That is the transductive case,
and it is what the text says. Not a refusal — nothing is silently frozen or
silently recomputed, which is the property this whole design exists to hold.

Skipped: the bare `sc(x)` sugar for global fit-transform. It expands to the
cross-join form above; add it when the spelling annoys someone.

## The gate

One law, because without marginalization there is no rewrite to verify:

```python
t.fit(D).transform(D) == run(t, D)
```

where `run(t, D)` executes `t`'s SQL directly against `D` with `memorize`
treated as identity. Freezing on the data you fitted on changes nothing.

It holds only at `D`, which is exactly what makes it a test of *what* got
frozen rather than a tautology — a transform that froze an ordered frame
fails it the moment `D` has a repeated order key.

Property-test it over the corpus. It is cheap: both sides are the same SQL
text modulo one function call.

## Slices

1. **`SQLTransform`, `memorize`, `fit`/`transform`.** Single level, no
   nesting. Parse `memorize`, check admissibility at construction (P7 —
   "refusals are construction-time and named"), freeze at fit, substitute at
   transform. Gate: the law above, plus `CannotMemorize` on the ordered
   frame.
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

Designed and then cut from this spec — reasoning is in commit `6a3b23a`, not
lost:

- **`SQLProjection`** as a second type, defined by row-locality of the
  residual. Needs marginalization to define, and needs `compile()` to be
  worth having.
- **Marginalization** — automatic lifting of windows into `memorize`.
  `memorize` is hand-written here.
- **`compile()` / `Inference`** — the serving artifact and the row path.
- **Scalar-position `memorize`** — `price / memorize(avg(price) OVER (...))`.
  Measured to parse and round-trip as a scalar node, so the mechanism is
  available; deferred because it is the one place key inference returns. This
  is the spelling people will reach for, and slice 1 ships without it.
