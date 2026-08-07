# Timeseries

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

A `.rolling()` call looks the same whether it is a live feature or a frozen
statistic; which one it is lives in the surrounding code. Here it is the
parameter the window is written over. That is a smaller difference than it
sounds — it moves the distinction into the text instead of removing it.

```
>>> from datetime import date
>>> TS = pa.table({
...     "series": ["a"] * 4 + ["b"] * 4,
...     "ts": pa.array([date(2024, 1, d) for d in (1, 2, 3, 4)] * 2, pa.date32()),
...     "value": [10.0, 12.0, 14.0, 16.0, 100.0, 110.0, 120.0, 130.0],
... })
>>> FUTURE = pa.table({
...     "series": ["a", "b", "c"],
...     "ts": pa.array([date(2024, 1, 5)] * 3, pa.date32()),
...     "value": [18.0, 140.0, 7.0],
... })

```

**Calendar features are stateless** — no `__FIT__`, so `fit` is a no-op and
there is nothing to ship:

```
>>> cal = SQLTransform('''
...     SELECT t.ts::VARCHAR AS ts, dayofweek(t.ts) AS dow
...     FROM __THIS__ t WHERE t.series = 'a' ORDER BY t.ts
... ''')
>>> cal.fit(TS).params
{}
>>> cal.fit(TS)(TS).to_pylist()[:2]
[{'ts': '2024-01-01', 'dow': 1}, {'ts': '2024-01-02', 'dow': 2}]

```

**A rolling feature over `__THIS__` is live** — recomputed from whatever batch
arrives, nothing frozen:

```
>>> rolling = SQLTransform('''
...     SELECT t.ts::VARCHAR AS ts,
...            round(avg(t.value) OVER (PARTITION BY t.series ORDER BY t.ts
...                                     ROWS 1 PRECEDING), 2) AS ma2
...     FROM __THIS__ t WHERE t.series = 'a' ORDER BY t.ts
... ''')
>>> rolling.fit(TS).params
{}
>>> rolling.fit(TS)(TS).to_pylist()[:3]
[{'ts': '2024-01-01', 'ma2': 10.0}, {'ts': '2024-01-02', 'ma2': 11.0}, {'ts': '2024-01-03', 'ma2': 13.0}]

```

**A level over `__FIT__` is frozen** — learned once, one row per series, and an
unseen series is a NULL rather than a crash:

```
>>> level = SQLTransform('''
...     SELECT t.series, t.ts::VARCHAR AS ts, round(t.value / m.level, 4) AS rel
...     FROM __THIS__ t
...     LEFT JOIN (SELECT series, avg(value) AS level FROM __FIT__ GROUP BY series) m
...       ON t.series = m.series
...     ORDER BY t.series, t.ts
... ''')
>>> {k: len(v) for k, v in level.fit(TS).params.items()}
{'__param_0': 2}
>>> level.fit(TS)(FUTURE).to_pylist()
[{'series': 'a', 'ts': '2024-01-05', 'rel': 1.3846}, {'series': 'b', 'ts': '2024-01-05', 'rel': 1.2174}, {'series': 'c', 'ts': '2024-01-05', 'rel': None}]

```

**Backtesting is the same object, fitted on less.** Fit on history, transform
the future — no separate code path, and no way for the two to disagree:

```
>>> HISTORY = TS.filter(pc.less(TS["ts"], pa.scalar(date(2024, 1, 4))))
>>> len(HISTORY)
6
>>> level.fit(HISTORY)(FUTURE).to_pylist()[:2]
[{'series': 'a', 'ts': '2024-01-05', 'rel': 1.5}, {'series': 'b', 'ts': '2024-01-05', 'rel': 1.2727}]

```

**Lag joined by key: read this one twice.** An ordered frame over `__FIT__` is
legal, and it means *join the training data on this key*. A serving timestamp
the training data never had gets NULL — forever, not just once:

```
>>> lagged = SQLTransform('''
...     SELECT t.ts::VARCHAR AS ts, h.prev
...     FROM __THIS__ t
...     LEFT JOIN (SELECT ts, series,
...                       lag(value) OVER (PARTITION BY series ORDER BY ts) AS prev
...                FROM __FIT__) h
...       ON t.ts = h.ts AND t.series = h.series
...     WHERE t.series = 'a' ORDER BY t.ts
... ''')
>>> lagged.fit(TS)(TS).to_pylist()[:3]
[{'ts': '2024-01-01', 'prev': None}, {'ts': '2024-01-02', 'prev': 10.0}, {'ts': '2024-01-03', 'prev': 12.0}]
>>> lagged.fit(TS)(FUTURE).to_pylist()
[{'ts': '2024-01-05', 'prev': None}]

```

It also ships the whole training set, because that is what it asked for — and
says so rather than hiding it:

```
>>> {k: len(v) for k, v in lagged.fit(TS).params.items()}
{'__param_0': 8}

```

For a lag computed from the *live* batch instead, put the window over
`__THIS__`. The two read almost identically and behave very differently, so
the model makes you name which one you meant. It does not stop you naming
the wrong one.
