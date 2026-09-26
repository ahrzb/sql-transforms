# Design: row-local derived tables and CTEs (2026-09-26)

**Status: proposed**, awaiting the owner's answers in the open-questions
section. Measured on master `e498019` against DuckDB 1.5.5 through
`confit.oracle.Oracle` (optimizer off).

## Summary

Confit will serve derived tables and CTEs whose body reads the request table
row-locally, by running the query as a pipeline of stages instead of inlining
the subquery. That recovers 150 of the 157 derived-table refusals of queries
DuckDB answers on the generated grammar (7.5% of the campaign). A second phase
serves CTEs and derived tables over static tables only (48 more,
projection-only) by computing them with confit's own engine at construction.

- **Why a pipeline, not inlining.** DuckDB evaluates every column of a
  subquery, including ones the outer query never reads, and finishes the
  subquery's SELECT before the outer WHERE runs. Substituting inner
  expressions into the outer query would drop traps DuckDB raises.
- **What changes inside confit.** The plan becomes a sequence of stages, and
  an outer stage reads the inner stage's results from slots. The binder binds
  a subquery with its existing machinery and hands the outer query its output
  columns. This rearchitects `lower.rs` and the binder's FROM handling; the
  public API does not change.
- **What stays refused.** Subqueries that aggregate, sort, limit or
  deduplicate over request rows; a CTE referenced twice; correlated, scalar and
  `IN` subqueries; set operations; CTEs that aggregate static tables.

## Scope

Refusals of queries DuckDB answers, campaign seeds 0–1999:

| subquery body | row-local | whole-relation | phase |
|---|---|---|---|
| reads the request table (all derived tables) | 150 | 7 | 1 |
| reads static tables only, projection | 48 | 3 | 2 |
| reads static tables only, aggregation | 40 | 3 | later, own ruling |

**Phase 1 serves**
- a derived table in `FROM` over the request table, aliased or not
  (`unnamed_subquery`), with a column-alias list (`AS t(p, q)`, partial lists
  too);
- a CTE referenced once whose body reads the request table (a named derived
  table);
- inside the subquery: `WHERE`, joins to static tables, struct field access,
  UDFs: everything the row path already serves;
- outside it: joins keyed on subquery columns, `WHERE`, `*` and `t.*`, nesting
  to any depth.

**Phase 2 serves** a derived table or CTE whose body reads static tables only,
without aggregation, joined to the request row path.

**Stays refused, by name**: aggregation, `GROUP BY`, `DISTINCT`, `ORDER BY`,
row limits or windows over request rows at any level; a CTE referenced twice;
correlated, scalar and `IN (SELECT …)` subqueries, `LATERAL`, set operations; a
static-only body that aggregates.

## DuckDB semantics to reproduce

| behavior | query (abridged) | DuckDB |
|---|---|---|
| every subquery column is evaluated, read or not | `SELECT a FROM (SELECT a, a + 1 AS b …)`, `a` = 2⁶³−1 | Overflow error |
| the subquery's SELECT finishes before the outer WHERE | `SELECT i FROM (SELECT CAST(s AS INTEGER) i, s …) WHERE s <> 'x'` | Conversion error |
| the subquery's own WHERE guards its SELECT | `SELECT CAST(s AS INTEGER) FROM (SELECT s … WHERE s <> 'x')` | serves |
| constants do not fold across levels | `SELECT k + 9223372036854775807 FROM (SELECT 1 AS k …)` | error per row; none on 0 rows |
| a bare NULL column stays SQLNULL through the level | `SELECT x \|\| 'a' FROM (SELECT NULL AS x …)` | NULL, typed INTEGER at output |
| unaliased subquery | `SELECT unnamed_subquery.x FROM (SELECT a AS x …)` | serves |
| column-alias list, partial | `(SELECT a, s …) AS t(p)` | columns `p`, `s` |
| duplicate names | `SELECT * FROM (SELECT a, a …)` | columns `a`, `a_1` |
| unnamed expressions | `SELECT * FROM (SELECT a + 1, upper(s) …)` | columns `(a + 1)`, `upper(s)` |
| inner scope is closed | outer `s` not projected; outer `__THIS__.a` | Binder errors |
| static-only CTE body traps | `WITH c AS (SELECT CAST(t AS INTEGER) … FROM d)`, 0 request rows | Conversion error on every query |
| CTE = named derived table | `WITH c AS (…) SELECT x FROM c` | same rows as the derived form |

## Architecture

A query becomes a pipeline of stages, one per query level. Each stage runs its
joins, then its WHERE, then evaluates every one of its SELECT columns into a
slot; the next level reads those slots as its input row.

- **Binder.** A subquery is bound by a fresh binder with the existing
  machinery. Its output columns, named by DuckDB's rules, become the outer
  binder's row table under the subquery's alias (`unnamed_subquery` when
  none). A column reference in the outer binder resolves to a slot instead of
  an input lane, so stars, `EXCLUDE`, qualification and ambiguity rules carry
  over unchanged. The inner scope is closed.
- **Plan.** `Rel` stops being a fixed `Project(Filter?(Scan))` and becomes a
  list of stages; each join records the stage it probes at.
- **Lowering.** `lower.rs` emits the stages in order. A stage's SELECT values
  ride the existing block-argument threading (`live`) into the next stage, so a
  slot is an SSA value computed once per row: one UDF call per row, no
  re-evaluation.
- **Reused unchanged:** the IR, both backends, the verifier, the boundary,
  static materialization and the key machinery.
- **Phase 2** compiles a static-only subquery with the static table as its
  input, runs it once over every static row at construction, and registers the
  output as a new static table. No DuckDB is involved.

## Correctness risks

| risk | how it is closed | pinned by |
|---|---|---|
| a trap in a column the outer query never reads goes missing | every inner SELECT column is evaluated into a slot | the overflow and cast cases above |
| an outer WHERE hides a trap the inner SELECT raises | stage order | the `CAST` / `WHERE s <> 'x'` case |
| an inner expression evaluated twice, or a UDF called twice | slots are SSA values computed once | a counting UDF referenced twice outside |
| a constant folded across a level | a slot is never foldable; bind-time folding stops at the level boundary | `k + 9223372036854775807` on 3 rows and on 0 rows |
| a bare NULL typed too early | phase 1 refuses an outer expression over a bare-NULL inner column, by name; direct projection serves typed INTEGER | the `x \|\| 'a'` case |
| `shape='map'` proved wrong | the one-row proof walks every stage | shape-contract tests per stage |
| a name resolves where DuckDB's binder errors | the outer binder sees only the subquery's output columns | the closed-scope cases |
| phase 2: a static body that traps serves anyway | the construction-time run traps, so construction refuses | the static `CAST` case on 0 and 2 request rows |

The generator already emits derived tables, so the campaign is the standing
differential for phase 1 from the first PR; it gains a CTE-over-request arm and
a trap-in-unread-column production.

## Delivery plan

| PR | what lands | what it serves |
|---|---|---|
| 1. Staged pipeline | `Rel` as stages, joins tagged with their stage, slots in lowering | nothing new; gate, campaign and corpus identical |
| 2. Derived tables | binder recursion, naming, alias lists, closed scope, the one-row proof over stages | row-local derived tables, one level, no joins inside |
| 3. Joins, nesting, CTEs | joins inside, outer joins on slots, any depth, a CTE referenced once | the rest of phase 1 |
| 4. Static-only subqueries | construction-time run over static rows | the 48 static projection subqueries |

A dated reading (N=4) follows PR 3.

## Open questions for the owner

1. Approve the staged-pipeline refactor (PR 1)?
2. Phase 2: static-only subqueries computed at construction by confit's own
   engine, a trap refusing construction. Approve, or keep them refused?
3. Out of scope unless the owner says otherwise: a CTE referenced twice,
   aggregating static subqueries, correlated/scalar/`IN` subqueries.
4. C1 depth: raise the default to 1,500 cases, or correct the text to 25?
