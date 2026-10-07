# Decorrelation

**Question:** Which correlated `__FIT__` subqueries can fit turn into params
tables without changing an answer? Which rewrites from the decorrelation
literature apply when fit and serving run at different times?

| Note | Contents |
|---|---|
| [Algorithms and traps](algorithms-and-traps.md) | Why fit and serving differ from query optimization, Kim's NEST-JA and its corrections, the modern algebra, what DuckDB does, the empty-group value, the rewrite and trap catalogues, and disclosure. |

What the findings meant for `sql_transform` is in the
[type-JA survey](../../records/research/decorrelation/type-ja-survey.md).
The current rules are in [decorrelation](../../spec/fit/decorrelation.md)
and [decorrelation refusals](../../spec/fit/decorrelation-refusals.md).
