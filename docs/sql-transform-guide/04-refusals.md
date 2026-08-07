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
