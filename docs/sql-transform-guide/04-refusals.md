# Where it refuses, and where it does not

<details>
<summary>Setup for the examples on this page</summary>

```
>>> import pyarrow as pa
>>> import pyarrow.compute as pc
>>> from datetime import date
>>> from sql_transform.model import SQLTransform, Transform, run

>>> SALES = pa.table({
...     "store": ["S1", "S1", "S1", "S2", "S2", "S2"],
...     "price": [10.0, 20.0, 30.0, 100.0, 300.0, 500.0],
... })

>>> z = SQLTransform("""
...     SELECT t.store, round(t.price / s.m, 4) AS z
...     FROM __THIS__ t, (SELECT avg(price) m FROM __FIT__) s
...     ORDER BY t.store, z
... """)

```

</details>

**Not refused:** a `__FIT__` subquery correlated into `__THIS__` by an
equality. It is per-serving-row, so it cannot be evaluated once as written —
but the group it asks for can, one row per key:

```
>>> per_store = SQLTransform('''
...     SELECT t.store, (SELECT avg(f.price) FROM __FIT__ f WHERE f.store = t.store) AS m
...     FROM __THIS__ t ORDER BY t.store
... ''')
>>> fitted = per_store.fit(SALES)
>>> {name: len(p) for name, p in fitted.params.items()}
{'__param_0': 2, '__param_1': 1}

```

Two stores, two rows, whatever the size of the training set — plus one row
holding what the subquery itself returns for a key that is not there, which
is the answer for a store fit never saw:

```
>>> fitted(pa.table({"store": ["S1", "NEW"]})).to_pylist()
[{'store': 'NEW', 'm': None}, {'store': 'S1', 'm': 20.0}]

```

**Refused**, at **construction**, when no `GROUP BY` reproduces the
correlation — here an inequality, which is real and common and temporary:

```
>>> SQLTransform('''
...     SELECT (SELECT avg(f.price) FROM __FIT__ f WHERE f.price <= t.price) AS m
...     FROM __THIS__ t
... ''')
Traceback (most recent call last):
    ...
sql_transform.model._errors.CorrelatedFit: __FIT__ subquery correlates out of itself and the correlation is not a conjunction of equalities — t.price joins the two relations some other way, so it cannot be evaluated once into a keyed table

```

`docs/decorrelation-unsupported.md` lists every shape that refuses this way,
each with what lifting it would take.

**Refused** too: anything that would put the whole training set in the
artifact without the text saying so.

```
>>> SQLTransform("SELECT t.price - f.price AS d FROM __THIS__ t, __FIT__ f")
Traceback (most recent call last):
    ...
sql_transform.model._errors.WholeTrainingSet: a bare `FROM __FIT__` beside __THIS__ would put the whole training set in the artifact. Wrap the __FIT__ reference in a subquery selecting the rows and columns you need — `(SELECT ... FROM __FIT__) f` — so the artifact's size is visible in the text

```

An unknown name refuses at construction too, rather than at serving time.

**Not refused, deliberately:** an ordered frame over `__FIT__`. It says *join
the training data on this key*, so a serving row with an unseen key gets NULL
forever. The hazard is real and it is visible in the text — which is why
refusing it would overrule something you can already see.
