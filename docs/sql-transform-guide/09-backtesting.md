# Backtesting: train on the last three months, score the next one

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

Twelve months, two stores, one trending up and one down:

```
>>> MONTHS = [date(2024, m, 1) for m in range(1, 13)]
>>> SALES12 = pa.table({
...     "store": ["S1"] * 12 + ["S2"] * 12,
...     "month": pa.array(MONTHS * 2, pa.date32()),
...     "price": [100.0 + 10 * i for i in range(12)]
...            + [500.0 - 20 * i for i in range(12)],
... })
>>> def window(lo, hi):
...     m = SALES12["month"]
...     return SALES12.filter(pc.and_(
...         pc.greater_equal(m, pa.scalar(MONTHS[lo], pa.date32())),
...         pc.less_equal(m, pa.scalar(MONTHS[hi], pa.date32())),
...     ))

```

### The loop

One transform, refitted per origin. `__FIT__` is literally the three months
you handed it, so *did this feature see the future* is a question you answer by
pointing, not by auditing:

```
>>> baseline = SQLTransform('''
...     SELECT t.month::VARCHAR AS month, t.store,
...            round(t.price / h.baseline, 4) AS rel
...     FROM __THIS__ t
...     LEFT JOIN (SELECT store, avg(price) AS baseline FROM __FIT__ GROUP BY store) h
...       ON t.store = h.store
...     ORDER BY t.month, t.store
... ''')

>>> rolling = []
>>> for i in range(3, 12):
...     rolling += baseline.fit(window(i - 3, i - 1)).transform(window(i, i)).to_pylist()
>>> len({r["month"] for r in rolling})
9
>>> rolling[:2]
[{'month': '2024-04-01', 'store': 'S1', 'rel': 1.1818}, {'month': '2024-04-01', 'store': 'S2', 'rel': 0.9167}]

```

Nine fits. Nothing is shared between them, and nothing has to be: the artifact
of one origin cannot leak into the next, because each is a separate `Fitted`.

### The same thing in one fit

An ordered frame over `__FIT__` *is* a rolling origin. `ROWS BETWEEN 3
PRECEDING AND 1 PRECEDING` excludes the row it is computed for, so the baseline
for a month never contains that month:

```
>>> one_shot = SQLTransform('''
...     SELECT t.month::VARCHAR AS month, t.store,
...            round(t.price / h.baseline, 4) AS rel
...     FROM __THIS__ t
...     LEFT JOIN (
...         SELECT store, month,
...                avg(price) OVER (PARTITION BY store ORDER BY month
...                                 ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS baseline
...         FROM __FIT__
...     ) h ON t.store = h.store AND t.month = h.month
...     ORDER BY t.month, t.store
... ''')
>>> shot = one_shot.fit(SALES12).transform(SALES12).to_pylist()
>>> shot[:4]
[{'month': '2024-01-01', 'store': 'S1', 'rel': None}, {'month': '2024-01-01', 'store': 'S2', 'rel': None}, {'month': '2024-02-01', 'store': 'S1', 'rel': 1.1}, {'month': '2024-02-01', 'store': 'S2', 'rel': 0.96}]

```

January has no history at all, so it is NULL rather than nothing. February and
March get a partial window. From April on, the window is full — and there the
two spellings are the same numbers:

```
>>> [r for r in shot if r["month"] >= "2024-04"] == rolling
True

```

Two implementations of the same idea, checked against each other in one line.
The windowed version is harder to read than the loop, and the loop is easier
to be sure of; the equality is what lets you pick on other grounds.

### Scoring every month

```
>>> mape = SQLTransform('''
...     SELECT t.month::VARCHAR AS month,
...            round(avg(abs((t.price / h.baseline) - 1)), 4) AS mape
...     FROM __THIS__ t
...     LEFT JOIN (
...         SELECT store, month,
...                avg(price) OVER (PARTITION BY store ORDER BY month
...                                 ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS baseline
...         FROM __FIT__
...     ) h ON t.store = h.store AND t.month = h.month
...     WHERE h.baseline IS NOT NULL
...     GROUP BY t.month ORDER BY t.month
... ''')
>>> mape.fit(SALES12).transform(SALES12).to_pylist()[:4]
[{'month': '2024-02-01', 'mape': 0.07}, {'month': '2024-03-01', 'mape': 0.102}, {'month': '2024-04-01', 'mape': 0.1326}, {'month': '2024-05-01', 'mape': 0.1268}]

```

### What each costs

The one-shot version ships a baseline per store-month, which is the whole
training grid:

```
>>> {k: len(v) for k, v in one_shot.fit(SALES12).params.items()}
{'__param_0': 24}

```

Nine fits and O(1) params each, or one fit and |D| params. Neither is hidden.

And the ordered frame's documented hazard is, in a backtest, exactly the right
answer — a month the training data never had has no baseline, so it is NULL:

```
>>> unseen = pa.table({
...     "store": ["S1"],
...     "month": pa.array([date(2025, 1, 1)], pa.date32()),
...     "price": [999.0],
... })
>>> one_shot.fit(SALES12).transform(unseen).to_pylist()
[{'month': '2025-01-01', 'store': 'S1', 'rel': None}]

```

In a backtest that NULL is correct: there is no baseline for that month. In
production it is the trap — an unseen key stays NULL for good, and nothing
warns you. The construct cannot tell which you are doing, so it is allowed
and you carry the risk.
