# The artifact's size is visible

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

A transform that genuinely needs the training set at serving time gets one —
and says so, instead of hiding it in a pickle. It has to *ask*, though: the
retained rows are a subquery you wrote, not a side effect of freezing.

```
>>> retains = SQLTransform('''
...     SELECT t.price - f.price AS d
...     FROM __THIS__ t, (SELECT price FROM __FIT__) f
... ''')
>>> {name: len(p) for name, p in retains.fit(SALES).params.items()}
{'__param_0': 6}

```

Six rows in, six rows of parameters. Compare `z` above: one row, whatever the
size of the training set.

Write `FROM __THIS__ t, __FIT__ f` instead and it refuses. Same artifact, but
its size would be a fact about the implementation rather than about the text —
and the subquery form is where you drop the columns you do not need.

A correlated `__FIT__` subquery costs one row per distinct key, not one per
training row:

```
>>> keyed = SQLTransform('''
...     SELECT t.store, (SELECT avg(f.price) FROM __FIT__ f WHERE f.store = t.store) AS m
...     FROM __THIS__ t
... ''')
>>> {name: len(p) for name, p in keyed.fit(SALES).params.items()}
{'__param_0': 2, '__param_1': 1}

```

Two stores in six rows, so two rows of parameters — and the extra one holds
what the subquery itself returns for a store fit never saw.
