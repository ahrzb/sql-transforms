# The surface

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
