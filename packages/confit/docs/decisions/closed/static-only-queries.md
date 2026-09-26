# Static-only queries: no frozen fold

**Question.** Should confit serve a query that reads no request table by
evaluating it once at construction and freezing the answer?

**Ruling.** No. A query that reads no request table is outside the model: it is
not a feature transform, and a caller with such a query runs it on DuckDB rather
than on confit. Confit defines no frozen-fold contract.

- **Scope.** The [goal](../../goal.md#scope) states the ground as batch dependence:
  the answer for one request must not depend on other requests. An aggregate over
  the static rows one input row matches stays inside the model as per-row
  aggregation.
- **Float reductions.** `sum` and `avg` over `DOUBLE` have no single oracle
  answer: on one measured table, twenty multi-threaded DuckDB runs produced twenty
  distinct bit patterns. That family is compared within the declared
  [float-reduction bound](../../oracle/05-the-comparison-contract.md#floating-point-comparisons).
  The oracle does not pin `threads = 1`.

## Ground: a fold is not a function of the query

A fold's answer can depend on three things beyond the query and the static tables:

| class | examples |
|---|---|
| execution order | tie order, `first`, `list`, `avg` over doubles, `LIMIT` without a total order, `ROWS` frames, `ASOF` and `POSITIONAL` joins, `DISTINCT ON` |
| session settings | `TimeZone`, `Calendar`, the optimizer |
| reads of the run | clocks, `random`, files, the catalogue, `version()`, `SUMMARIZE` |

Refusing such shapes by name does not converge: the set includes shapes that carry
no name a parse reading can see (an opaque `SHOW_REF` node, a file path in a table
position, a time zone applied by a plain cast, a zoned type chosen at bind time from
a string argument), and a missed shape serves a wrong constant silently. Pinning the
fold configuration removes the first two classes but makes frozen answers a function
of the fold configuration rather than of the query. Folding twice under two
configurations and refusing on disagreement is a probability, not a proof, and
cannot see reads of the run.

## The engine today

The engine still serves a static-tables-only query: `eval_static_only` in
`packages/confit/src/duckdb/mod.rs` evaluates it once at construction and freezes
the rows (claim: one-door-bypass in [the oracle](../../oracle/01-what-the-oracle-is.md)).
A row-limit clause on such a query and `shape='map'` are refused. Removing the path
is tracked in `packages/confit/PLANS.md` ("Remove the static-only fold").
