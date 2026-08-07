# SQLTransform: a guide

> A transform is a function `(F, T) -> R` over relations.

`__FIT__` and `__THIS__` are its two parameters. `fit` binds one, `transform`
the other. Which half is learned and which is live is read off the text — there
is no annotation to remember and none to forget.

**Every example below is executed.** `_docs_test.py` runs this file as a
doctest, so an output that drifts fails the suite.

```
>>> import pyarrow as pa
>>> import pyarrow.compute as pc
>>> from sql_transform.model import SQLTransform, Transform, run

>>> SALES = pa.table({
...     "store": ["S1", "S1", "S1", "S2", "S2", "S2"],
...     "price": [10.0, 20.0, 30.0, 100.0, 300.0, 500.0],
... })

```

## Contents

1. [The surface](01-the-surface.md)
1. [What the two-parameter form changes](02-what-the-two-parameter-form-changes.md)
1. [Patterns](03-patterns.md)
1. [Where it refuses, and where it does not](04-refusals.md)
1. [The artifact's size is visible](05-the-artifacts-size.md)
1. [It is an sklearn estimator](06-as-an-sklearn-estimator.md)
1. [Connections and unexecuted output](07-connections-and-lazy-output.md)
1. [Timeseries](08-timeseries.md)
1. [Backtesting: train on the last three months, score the next one](09-backtesting.md)
1. [Trade-offs](10-trade-offs.md)
1. [Not here yet](11-not-here-yet.md)

