# SQLTransform: a guide

> A transform is a function `(F, T) -> R` over relations.

`__FIT__` and `__THIS__` are its two parameters. `fit` binds one, `transform`
the other. Which half is learned and which is live is read off the text — there
is no annotation to remember and none to forget.

**Every example below is executed.** `_docs_test.py` runs this file as a
doctest, so an output that drifts fails the suite.

```
>>> import pyarrow as pa
>>> from sql_transform.model import SQLTransform, Transform, run

>>> SALES = pa.table({
...     "store": ["S1", "S1", "S1", "S2", "S2", "S2"],
...     "price": [10.0, 20.0, 30.0, 100.0, 300.0, 500.0],
... })

```

## The surface

```
>>> z = SQLTransform('''
...     SELECT t.store, round(t.price / s.m, 4) AS z
...     FROM __THIS__ t, (SELECT avg(price) m FROM __FIT__) s
...     ORDER BY t.store, z
... ''')

>>> fitted = z.fit(SALES)              # or z(SALES) — fit is partial application
>>> fitted(SALES).to_pylist()[:3]      # or fitted.transform(SALES)
[{'store': 'S1', 'z': 0.0625}, {'store': 'S1', 'z': 0.125}, {'store': 'S1', 'z': 0.1875}]

```

The artifact is **data**, not a pickle:

```
>>> fitted.params
{'__param_0': pyarrow.Table
m: double
----
m: [[160]]}

>>> print(fitted.sql)
SELECT t.store, round((t.price / s.m), 4) AS z FROM __THIS__ AS t , (SELECT * FROM __param_0) AS s ORDER BY t.store, z

```

Nothing in the residual reads `__FIT__`. Serving needs the params table and
nothing else — the training set is gone, and you can prove it:

```
>>> sum(len(p) for p in fitted.params.values())
1

```

## Why not just write a sklearn transformer

### 1. Train/serve skew becomes unrepresentable

The classic failure is two pieces of code — a training job and a serving path —
that are supposed to compute the same thing and slowly stop doing so.

```python
# traditional: the contract lives in your head, in two places
class Normalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_ = X["price"].mean()      # one expression here...
        return self
    def transform(self, X):
        return X["price"] / self.mean_      # ...and its partner here
```

Here both halves are one text, and `run` — which binds *both* parameters to the
same relation, with no freezing at all — is the reference the pair must match:

```
>>> a = run(z, SALES).to_pylist()
>>> b = z.fit(SALES).transform(SALES).to_pylist()
>>> a == b
True

```

That equality is a law, not a habit: `docs/properties.md` calls it *freezing is
faithful*.

### 2. Applying a groupless transform per group

`z` above knows nothing about stores. Making it per-store is a `LATERAL` and a
`WHERE` — the member is not modified, wrapped, or subclassed:

```
>>> per_store = SQLTransform('''
...     SELECT x.* FROM (SELECT DISTINCT store FROM __FIT__) g,
...       LATERAL z((FROM __FIT__  WHERE store = g.store),
...                 (FROM __THIS__ WHERE store = g.store)) x
...     ORDER BY x.store, x.z
... ''')
>>> per_store.fit(SALES)(SALES).to_pylist()
[{'store': 'S1', 'z': 0.5}, {'store': 'S1', 'z': 1.0}, {'store': 'S1', 'z': 1.5}, {'store': 'S2', 'z': 0.3333}, {'store': 'S2', 'z': 1.0}, {'store': 'S2', 'z': 1.6667}]

```

The traditional shape is a loop, a dict of fitted estimators, and a lookup you
have to remember to make total:

```python
# traditional: three places to get wrong, one of which only fails in production
models = {g: clone(scaler).fit(part) for g, part in X.groupby("store")}
...
models[row.store]            # KeyError on a store that appeared after training
```

**Both parameters must be sliced.** Filtering only `__FIT__` leaves `__THIS__`
whole, so every group crosses every row — measured, 4 rows in and 8 out, with
S1's statistics applied to S2's rows.

### 3. Joins are first-class

A transformer that needs a lookup table has nowhere to put it; the join happens
outside the pipeline and is not part of the artifact. Here it is just SQL, and
the table is captured **by value** at construction:

```
>>> REGION = pa.table({"store": ["S1", "S2"], "region": ["north", "south"]})
>>> with_region = SQLTransform('''
...     SELECT r.region, round(avg(t.price), 2) AS avg_price
...     FROM __THIS__ t JOIN REGION r ON t.store = r.store
...     GROUP BY r.region ORDER BY r.region
... ''')
>>> with_region.fit(SALES)(SALES).to_pylist()
[{'region': 'north', 'avg_price': 20.0}, {'region': 'south', 'avg_price': 300.0}]

>>> list(with_region.bindings)
['REGION']

```

Rebinding `REGION` afterwards does not change what was built.

### 4. Chaining says which data each stage learned from

`Pipeline` refits each stage on the previous stage's *transformed training
data*. It does that invisibly. Here each stage appears twice, once per
parameter, so the thing that is easy to get wrong is the thing you can see:

```
>>> shift = SQLTransform('''
...     SELECT t.store, t.price - s.lo AS price
...     FROM __THIS__ t, (SELECT min(price) lo FROM __FIT__) s
... ''')
>>> chained = SQLTransform('''
...     SELECT * FROM z(shift(__FIT__, __FIT__),      -- z learns from shifted training data
...                     shift(__FIT__, __THIS__)) s   -- and is applied to shifted live data
...     ORDER BY s.store, s.z
... ''')
>>> chained.fit(SALES)(SALES).to_pylist()[:3]
[{'store': 'S1', 'z': 0.0}, {'store': 'S1', 'z': 0.0667}, {'store': 'S1', 'z': 0.1333}]

```

Fitting `z` on the *raw* training data instead would give `0.0625` and `0.125`
— close enough to never notice, far enough to matter.

### 5. Unseen categories have exactly one story

A join miss is a NULL. Not a `KeyError`, not a silent zero, not a dropped row:

```
>>> encoder = SQLTransform('''
...     SELECT t.store, m.avg_price
...     FROM __THIS__ t
...     LEFT JOIN (SELECT store, avg(price) AS avg_price FROM __FIT__ GROUP BY store) m
...       ON t.store = m.store
...     ORDER BY t.store
... ''')
>>> new_batch = pa.table({"store": ["S1", "S9"], "price": [11.0, 99.0]})
>>> encoder.fit(SALES)(new_batch).to_pylist()
[{'store': 'S1', 'avg_price': 20.0}, {'store': 'S9', 'avg_price': None}]

```

The row survives. `docs/properties.md` calls this P14, the one NULL story.

### 6. sklearn where you actually want sklearn

An estimator is already the `(fit, transform)` pair, so it drops in as a leaf —
per group, without a loop:

```
>>> from sklearn.preprocessing import StandardScaler
>>> sc = Transform.from_estimator(StandardScaler(), takes=("v",), returns=("v",))
>>> scaled = SQLTransform('''
...     SELECT t.store, round(sc_transform(f.theta, struct_pack(v := t.price)).v, 4) AS z
...     FROM __THIS__ t
...     LEFT JOIN (SELECT store, sc_fit(struct_pack(v := price)) AS theta
...                FROM __FIT__ GROUP BY store) f
...       ON t.store = f.store
...     ORDER BY t.store, z
... ''')
>>> scaled.fit(SALES)(SALES).to_pylist()[:3]
[{'store': 'S1', 'z': -1.2247}, {'store': 'S1', 'z': 0.0}, {'store': 'S1', 'z': 1.2247}]

```

θ is an opaque handle into a registry, so the params table stays inspectable
while a fitted `RandomForest` is a pointer:

```
>>> f = scaled.fit(SALES)
>>> sorted(t["type"] for t in list(f.params.values())[0].column("theta").to_pylist())
['sc', 'sc']
>>> len(f.instances)
2

```

## Patterns

| you want | write |
| --- | --- |
| learn from training, apply to live | `FROM __THIS__ t, (SELECT ... FROM __FIT__) s` |
| the same, per group | `LATERAL m((FROM __FIT__ WHERE k = g.k), (FROM __THIS__ WHERE k = g.k))` |
| reuse a transform | `FROM m(__FIT__, __THIS__)` |
| chain two | `FROM b(a(__FIT__, __FIT__), a(__FIT__, __THIS__))` |
| a static lookup | name it; it resolves from your frame and is captured by value |
| an sklearn estimator | `Transform.from_estimator(est, takes=..., returns=...)`, then `x_fit`/`x_transform` |
| refit on every batch, deliberately | `x_fit(...) FROM __THIS__` — you had to type `__THIS__` |

## Where it refuses, and where it does not

One refusal: a `__FIT__` subtree may not correlate into `__THIS__`, because
that is per-serving-row and cannot be evaluated once. It raises at
**construction**, naming the column:

```
>>> SQLTransform('''
...     SELECT (SELECT avg(price) FROM __FIT__ f WHERE f.store = t.store) AS m
...     FROM __THIS__ t
... ''')
Traceback (most recent call last):
    ...
sql_transform.model._transform.CorrelatedFit: __FIT__ subquery references t.store from the outer query, so it cannot be evaluated once into a table

```

An unknown name refuses at construction too, rather than at serving time.

**Not refused, deliberately:** an ordered frame over `__FIT__`. It says *join
the training data on this key*, so a serving row with an unseen key gets NULL
forever. The hazard is real and it is visible in the text — which is why
refusing it would overrule something you can already see.

## Cost is a measurement, not a promise

A transform that genuinely needs the training set at serving time gets one —
and says so, instead of hiding it in a pickle:

```
>>> retains = SQLTransform("SELECT t.price - f.price AS d FROM __THIS__ t, __FIT__ f")
>>> {name: len(p) for name, p in retains.fit(SALES).params.items()}
{'__param_fit': 6}

```

Six rows in, six rows of parameters. Compare `z` above: one row, whatever the
size of the training set.

## Not here yet

Marginalization (so per-group fitting collapses to one row per group instead of
retaining the training set), correlated `__FIT__` subqueries, and the
row-at-a-time serving artifact. See
`docs/superpowers/specs/2026-08-07-datamodel-redesign-design.md`.
