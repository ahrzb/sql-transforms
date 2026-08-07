# SQLTransform, memorize, nesting (v0)

Date: 2026-08-07. Status: designed with AmirHossein (this session).
Scope deliberately small: one type, one freezing mechanism, one nesting rule.
Marginalization and the serving path are deferred — see the end.

## The type

```python
class SQLTransform:
    def __init__(self, sql, *, transforms=None, tables=None): ...
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

## Nesting

A transform is usable inside another transform as a table function:

```python
z = SQLTransform("... LEFT JOIN memorize(cat_stats) m USING (cat)")
outer = SQLTransform("SELECT * FROM z(__THIS__) WHERE z > 0", transforms={"z": z})
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
2. **Nesting.** `transforms={...}`, splice, name-prefix, depth cap. Gate: the
   law on a nested transform, plus spliced text equals a hand-written
   reference.

TDD, red before green. Lands alongside the existing modules in
`sql_transform/model/`; nothing is deleted until the old implementation and
the new one agree on the corpus.

## Blocking question

**Does `memorize(cat_stats)` survive DuckDB's parser?** We rewrite before
execution so it never has to be a registered function, but it must parse.
Cheap to check; blocks slice 1.

## Deferred

Designed and then cut from this spec — reasoning is in commit `6a3b23a`, not
lost:

- **`SQLProjection`** as a second type, defined by row-locality of the
  residual. Needs marginalization to define, and needs `compile()` to be
  worth having.
- **Marginalization** — automatic lifting of windows into `memorize`.
  `memorize` is hand-written here.
- **`compile()` / `Inference`** — the serving artifact and the row path.
- **sklearn leaves** as UDAF/UDF pairs, and opaque `theta` handles.
- **Scalar-position `memorize`** — `price / memorize(avg(price) OVER (...))`.
  The ergonomic spelling, and the one place key inference returns.
