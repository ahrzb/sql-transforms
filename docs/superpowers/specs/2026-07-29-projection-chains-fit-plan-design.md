# Projection chains, scalar subqueries, and the fit plan

**Date:** 2026-07-29
**Status:** loop 3 of SQLProjection marginalization; extends
[loop 1](2026-07-29-sql-projection-marginalization-design.md) and
[loop 2](2026-07-29-window-widening-design.md).

## The fit plan (AmirHossein's DAG point)

Once one aggregation can depend on another — a CTE that centers a column, a
second level that aggregates the centered values — fit execution is a DAG,
not a pair of queries. `Marginalized` therefore now carries an explicit,
topologically ordered plan:

```python
@dataclass(frozen=True)
class FitStep:
    name: str                # the table this step registers
    sql: str                 # runs against previously registered names
    reads: tuple[str, ...]   # the tables it consumes (the DAG edges)

Marginalized(serving_sql, plan: tuple[FitStep, ...], params: tuple[ParamsSpec, ...])
```

`fit()` is now a five-line executor: register `__THIS__`, run each step in
order, register its result under `step.name`, keep the params tables at the
end. Every intermediate is inspectable by name, every step is a plain SQL
string you can run by hand, and `SQLProjection.plan` exposes the whole thing —
debuggability is the point, not a by-product. `ParamsSpec` shrinks to
`(name, keys)`; the SQL lives in the plan.

Step naming: `__CF_LEVEL_{i}__` for a projection level's windows table,
`__CF_PARAMS_{n}__` for every params table (keyset collapses and scalar
subqueries alike).

## Projection chains: CTEs and derived tables

A query is now a **chain of strict projections**: the final SELECT over a
CTE or derived table, which projects over another, down to `__THIS__`. Each
level obeys the loop-1 structural rules (no WHERE/GROUP BY/modifiers) and the
loop-2 window rules, applied per level.

- **Fit** materializes each level: `__CF_LEVEL_{i}__` = the level's original
  select items under their original output names (the next level's input,
  verbatim) *plus* that level's distinct windows as `__cf_wN` and key
  expressions as `__cf_kM`, reading from the previous level's table. Params
  collapse per keyset exactly as loop 2 — the same query serves as both the
  relation and the windows table.
- **Serving** flattens: each level's column references are substituted with
  the *rewritten* expressions of the level below (β-reduction), so
  `serving_sql` is one flat projection over `__THIS__` joined to every params
  table. A level-2 params join can reference level-1 params columns in its
  key expression — the left-deep join chain is in creation order, so
  everything referenced is already in scope. Correctness composes by
  induction: each level's round-trip invariant says its rewritten form equals
  its original form on the training set.
- CTE column aliases (`WITH a(z) AS …`) and derived-table aliases rename the
  level's outputs; the source's name/alias is a valid qualifier in the
  consumer.
- Refused, named: recursive CTEs, unused CTEs, a CTE referenced by a scalar
  subquery, duplicate output names in a non-final level, `*` with
  EXCLUDE/REPLACE in a non-final level, struct-field access through a
  projected expression. (`MATERIALIZED` hints do not survive DuckDB's
  serializer, so they cannot be — and are not — distinguished.)
- A bare star in a non-final level passes base columns through; a star in the
  final level over a CTE expands to the source's outputs.

**Lateral-alias safety (latent loop-2 hole, fixed):** DuckDB lets a select
item reference an earlier item's alias, and prefers the table column when
both exist — undecidable without a schema. A bare reference that matches an
earlier alias in the same level is refused with a disambiguation hint
(qualify as `__THIS__.x` or rename the alias).

## Scalar subqueries: the strongest form of "run the original"

`FROM __THIS__` may carry no alias and no other relations are in scope, so a
subquery inside a projection level **has no syntax to reference the outer
row** — every subquery is provably uncorrelated. Any `SCALAR` or `EXISTS`
subquery whose base tables are all `__THIS__` (its own internal CTEs
included; WHERE/GROUP BY/anything inside is fine — it is a scalar
computation, not a projection) therefore becomes one fit step run
**verbatim**:

```sql
-- SELECT x / (SELECT max(x) FROM __THIS__) AS xn FROM __THIS__   becomes
plan:    __CF_PARAMS_0__ = SELECT (SELECT max(x) FROM __THIS__) AS __cf_a0
serving: SELECT (__cf_t.x / __cf_p0.__cf_a0) AS xn FROM __THIS__ AS __cf_t
         LEFT JOIN __CF_PARAMS_0__ AS __cf_p0 ON ((1 = 1))
```

Verbatim execution is bit-exact by construction — there is nothing to
re-derive. `IN`/`ANY`/`ALL` subqueries stay refused (their value varies per
row: a membership *function*, not a scalar). Prepared-statement parameters
(`$1`) get a named refusal instead of a confusing fit-time error.

## The progression metric: corpus replay

Loop 3 adds the house metric — a corpus replayed through three outcomes,
zero failures required, with **pinned counts** whose diffs over time are the
progression record (Confit's 550/678, for marginalization):

- **MARGINALIZED** — `marginalize` accepts and the training-set round-trip
  invariant holds bit-exactly.
- **REFUSED** — a named `MarginalizeError`.
- **FAILED** — anything else: a gate mismatch, an unexpected exception. The
  corpus test asserts this list is empty.

The corpus has two halves: window queries **mined from DuckDB's own test
suite** (`duckdb/test/sql/window/*.test`, the `empsalary` queries, replayed
against the suite's own ten rows; query-level ORDER BY — test scaffolding,
refused by design — is stripped during mining), and a curated set covering
every supported and refused family from all three loops. The scoreboard is a
pinned `{"marginalized": N, "refused": M}` per half; widening a future loop
means editing those pins upward in a reviewable diff.
