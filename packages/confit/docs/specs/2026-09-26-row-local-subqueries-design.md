# Design: row-local derived tables and CTEs (2026-09-26)

**Status: approved with conditions (2026-09-27).** The owner approved the staged
pipeline, phase 2 subject to the binding-schema and construction-refusal
conditions below, the listed exclusions, and a C1 default depth of 1,500. This
revision folds in the owner's review; the [review record](#review-record) lists
each change. Measured on master `e498019` against DuckDB 1.5.5 through
`confit.oracle.Oracle` (optimizer off).

## Summary

Confit will serve derived tables and CTEs whose body reads the request table
row-locally, by running the query as a pipeline of stages instead of inlining
the subquery. Phase 1 targets 150 row-local derived-table candidates (of 157
derived-table refusals of queries DuckDB answers, 7.5% of the campaign);
verified recovery is measured after PR 3. Phase 2 targets 48 candidates whose
body reads static tables only, projection-only, computed by confit's own engine
at construction; verified recovery is measured after PR 4.

- **Why a pipeline, not inlining.** DuckDB evaluates every column of a
  subquery, including ones the outer query never reads, and finishes the
  subquery's SELECT before the outer WHERE runs. Substituting inner
  expressions into the outer query would drop traps DuckDB raises.
- **What changes inside confit.** The plan becomes a sequence of stages. Each
  stage owns its joins, predicate and projection; an outer stage reads the
  inner stage's results from slots. The binder binds a subquery with its
  existing machinery and hands the outer query its output columns as a logical
  binding schema. This rearchitects `lower.rs` and the binder's FROM handling;
  the public API does not change.
- **What stays refused.** Subqueries that aggregate, sort, limit or
  deduplicate over request rows; a CTE referenced twice; correlated, scalar and
  `IN` subqueries; set operations; CTEs that aggregate static tables.

## Scope

Refusals of queries DuckDB answers, campaign seeds 0–1999. "Row-local" means
the case contains no whole-relation construct; it makes a case a candidate, not
a case that will serve. A leading derived-table refusal can hide a later one: an
unsupported expression or output type, a shape constraint, join multiplicity,
or the SQLNULL rule below.

| subquery body | row-local candidates | whole-relation | phase |
|---|---|---|---|
| reads the request table (all derived tables) | 150 | 7 | 1 |
| reads static tables only, projection | 48 | 3 | 2 |
| reads static tables only, aggregation | 40 | 3 | later, own ruling |

**Candidate snapshot.** Before any generator change, the 150 and 48 candidates
are frozen as a corpus (SQL, schemas, statics and request rows per case) under
`fuzz/corpora/`. Recovery is measured on that snapshot, so new generator
productions, which shift the seed stream, cannot change the denominator.

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
without aggregation, joined to the request row path. The query as a whole still
reads the request table; this is not the removed whole-query fold
([static-only queries](../decisions/closed/static-only-queries.md)).

**Stays refused, by name**: aggregation, `GROUP BY`, `DISTINCT`, `ORDER BY`,
row limits or windows over request rows at any level; a CTE referenced twice;
correlated, scalar and `IN (SELECT …)` subqueries, `LATERAL`, set operations; a
static-only body that aggregates; under `shape='many'`, more than one join in
the whole pipeline.

## DuckDB semantics to reproduce

| behavior | query (abridged) | DuckDB |
|---|---|---|
| every subquery column is evaluated, read or not | `SELECT a FROM (SELECT a, a + 1 AS b …)`, `a` = 2⁶³−1 | Overflow error |
| the subquery's SELECT finishes before the outer WHERE | `SELECT i FROM (SELECT CAST(s AS INTEGER) i, s …) WHERE s <> 'x'` | Conversion error |
| the subquery's own WHERE guards its SELECT | `SELECT CAST(s AS INTEGER) FROM (SELECT s … WHERE s <> 'x')` | serves |
| constants do not fold across levels | `SELECT k + 9223372036854775807 FROM (SELECT 1 AS k …)` | error per row; none on 0 rows |
| a bare NULL column stays SQLNULL through the level | `SELECT x \|\| 'a' FROM (SELECT NULL AS x …)` | NULL, typed INTEGER at output |
| … and materializing it loses that | `SELECT c.x \|\| 'a' FROM __THIS__ r JOIN (SELECT a, NULL AS x FROM d) c ON r.a = c.a` | INTEGER as a derived table; VARCHAR once `c` is materialized and registered |
| unaliased subquery | `SELECT unnamed_subquery.x FROM (SELECT a AS x …)` | serves |
| column-alias list, partial | `(SELECT a, s …) AS t(p)` | columns `p`, `s` |
| duplicate names | `SELECT * FROM (SELECT a, a …)` | columns `a`, `a_1` |
| unnamed expressions | `SELECT * FROM (SELECT a + 1, upper(s) …)` | columns `(a + 1)`, `upper(s)` |
| inner scope is closed | outer `s` not projected; outer `__THIS__.a` | Binder errors |
| static-only CTE body traps | `WITH c AS (SELECT CAST(t AS INTEGER) … FROM d)`, 0 request rows | Conversion error at execution, on every query |
| CTE = named derived table | `WITH c AS (…) SELECT x FROM c` | same rows as the derived form |

## Architecture

A query becomes a pipeline of stages, one per query level. Each stage runs its
joins, then its WHERE, then evaluates every one of its SELECT columns into a
slot; the next level reads those slots as its input row.

### Plan: stages own their work

`Rel` stops being a fixed `Project(Filter?(Scan))` and becomes an ordered list
of stages. A stage owns its ordered joins, its predicate and its projection;
there is no separate query-wide join list tagged with stage numbers, so no
invariant ties a tag to a position. Where static storage needs a query-wide
index (the static table a join probes, `model_base` in `lower.rs`), that index
is **storage identity**, allocated by the query context below; which stage runs
the probe is **execution ownership**, given by the join's place in its stage.
The two are separate fields and neither is derived from the other.

### Binder: closed scope, shared resources

Binding splits into two parts.

- **Level scope (closed).** Each query level binds against its own scope: the
  relations in its `FROM` and, for an outer level, the subquery's output
  columns under its alias (`unnamed_subquery` when none), named by DuckDB's
  rules. An inner name is invisible outside; an outer name is invisible inside.
  Stars, `EXCLUDE`, qualification and ambiguity rules carry over unchanged,
  because a slot is resolved as a column of the subquery's relation.
- **Query context (shared, append-only).** One per query, passed to every
  level's binder:
  - input lanes: the request-table scan exists at exactly one stage, and every
    lane minted for it, wherever the reference occurs, lands in the one
    query-wide input schema;
  - static tables and join storage IDs, allocated query-wide so two levels
    joining the same static table get distinct IDs where their keys differ;
  - regex IDs and model references, query-wide, so the compiled pattern table
    and model registry stay one list each;
  - UDFs: the one registry the query was built with.
  Nothing in the context is remapped between levels; IDs are unique by
  construction, so a stage's IR refers to them directly.

### Binding schema versus storage schema

A subquery's output has two schemas, kept as separate fields:

- the **binding schema**: each column's logical type exactly as the body bound
  it, SQLNULL included; the outer level binds only against this;
- the **storage schema**: the physical representation of a slot (phase 1) or a
  materialized static column (phase 2). A SQLNULL column is stored as an
  all-NULL nullable column of the width DuckDB would give it.

The outer binding schema is never reconstructed from storage. In both phases an
outer expression whose operand is a SQLNULL column is refused by name
(`unsupported: expression over a bare NULL subquery column`) until SQLNULL
propagation through operators is implemented; projecting such a column directly
serves, typed as DuckDB types it. One rule, both phases: the materialization
row in the semantics table is its phase-2 pin.

### Lowering: computed once per row reaching the stage

`lower.rs` emits the stages in order. A stage's SELECT values ride the existing
block-argument threading (`live`) into the next stage, so a slot is an SSA value
computed **once per row that reaches that stage**, after the stage's joins and
predicate. A row an inner WHERE drops never evaluates that stage's SELECT, and
a row a stage's join fans out (under `shape='many'`) evaluates it once per
output row. It is not once per original request row.

### `shape='many'`

The one-join restriction stays global across the pipeline: at most one join in
the whole query, at any stage (the guard at `lower.rs` becomes a count over all
stages). A join at one stage under `many` followed by further stages lowers as
the existing loop with the later stages inside its body. Joins at two stages
under `many` would compose multiplicities and are out of scope.

### Phase 2: static-derived relations

A static-only subquery is bound like any subquery, compiled with the static
table as its input, run once over every static row at construction, and
registered as a static table carrying both schemas above. No DuckDB is
involved. Static materialization and the key machinery are reused, with one
extension: a registered static table carries a binding schema, and the binder
reads it in preference to the Arrow schema.

**Construction-time trap: a refusal policy.** If the construction-time run
traps, confit refuses to construct the query: *confit refuses to construct this
static-derived relation because its construction-time evaluation traps.* This
is not execution equivalence with DuckDB, which accepts the query and raises at
execution. It is a named refusal:

- message: `unsupported: static-derived relation <name> traps during
  construction-time evaluation: <error>`;
- campaign verdict: `REFUSED`, with DuckDB's runtime error kept as the
  baseline outcome, as for every refusal;
- contract: PR 4 adds a named claim to
  [verdicts, agreement, abstention and refusal](../oracle/04-verdicts-agreement-abstention-refusal.md#construction-refusal-versus-runtime-trap)
  stating this policy as the one place where a construction-time evaluation
  refuses construction, and the serving contract says the same. The rule that
  a runtime trap is never rewritten as a construction refusal stays in force for
  everything evaluated per request.

The ground for the policy: the relation's content is fixed at construction, so
the trap is certain on every request; serving it would mean serving a program
that traps on every row.

### Reused unchanged

The IR, both backends, the verifier and the boundary.

## Correctness risks

| risk | how it is closed | pinned by |
|---|---|---|
| a trap in a column the outer query never reads goes missing | every inner SELECT column is evaluated into a slot | the overflow and cast cases above |
| an outer WHERE hides a trap the inner SELECT raises | stage order | the `CAST` / `WHERE s <> 'x'` case |
| an inner expression evaluated twice, or a UDF called twice | slots are SSA values computed once per row reaching the stage | a counting UDF referenced twice outside |
| a constant folded across a level | a slot is never foldable; bind-time folding stops at the level boundary | `k + 9223372036854775807` on 3 rows and on 0 rows |
| a bare NULL typed too early, or re-typed by materialization | binding schema kept apart from storage; the named SQLNULL refusal in both phases | the `x \|\| 'a'` case, derived and phase-2 forms |
| `shape='map'` proved wrong | the one-row proof walks every stage | shape-contract tests per stage |
| `shape='many'` multiplicity composed | one join per pipeline | two joins at different stages refuse |
| a name resolves where DuckDB's binder errors | each level binds against its closed scope | the closed-scope cases |
| two levels collide on a regex, join or model ID | IDs allocated by the shared query context | the same static table joined at two levels |
| phase 2: a static body that traps serves anyway | construction-time run; the named refusal | the static `CAST` case on 0 and 2 request rows |

## Acceptance gate

Each PR's campaign acceptance, seeds 0–1999 and the candidate snapshot:

- **zero** `DIVERGE_VALUE`, `DIVERGE_TRAP`, `DIVERGE_BUILD` and `OPT_EMULATED`;
- **no unexplained** `BUILD_EXC`, `PANIC`, `TIMEOUT` or `SKIP`: each one present
  is named in the PR with its cause;
- `DIVERGE_OPT` reported separately: it is disagreement with the optimizer-on
  reading, not with the chosen oracle, and does not gate;
- **case-level preservation**: every seed that is `AGREE` on master is `AGREE`
  after the change. Equal totals are not evidence. PR 1 must show the per-seed
  verdicts identical to master's.

The runner enforces it rather than a reader: before PR 1 it gains a per-seed
verdict file (`--cases`), a comparison against a baseline file
(`--baseline`), and a non-zero exit status when a gated verdict appears or a
baseline `AGREE` is lost. The trap-focused pins above stay in the gate
alongside it.

## Delivery plan

| PR | what lands | what it serves |
|---|---|---|
| 0. Runner and snapshot | per-seed verdicts, baseline comparison and exit status; the candidate snapshot | nothing new |
| 1. Staged pipeline | `Rel` as stages owning their joins, predicate and projection; slots in lowering | nothing new; per-seed verdicts, gate and corpus identical |
| 2. Derived tables | the query context, binder recursion, naming, alias lists, closed scope, the SQLNULL refusal, the one-row proof over stages | row-local derived tables, one level, no joins inside |
| 3. Joins, nesting, CTEs | joins inside, outer joins on slots, any depth, a CTE referenced once, the `many` global join count | the rest of phase 1; recovery on the snapshot measured |
| 4. Static-only subqueries | binding schema on registered statics, the construction-time run, the refusal claim in the contract | phase 2; recovery on the snapshot measured |

A dated reading (N=4) follows PR 3.

## Measured recovery (2026-09-27)

On the frozen candidate snapshot (`fuzz/corpora/subquery-candidates-2026-09-27.jsonl`):

| after | phase 1 (150) | phase 2 (48) |
|---|---|---|
| PR 2, derived tables | 132 agree | 0 |
| PR 3a, joins beside them and CTEs | 132 agree | 42 agree |

Phase 1's other 18 hit refusal classes the derived-table refusal was hiding
(the i128 literal, UDF widths, `shape='map'` with a WHERE, a failing constant
cast, a decimal literal).

**Phase 2 was mis-measured.** All 48 of its candidates are CTEs the query
never reads: the generator's CTE arm drops the join when it cannot form an
equality, and DuckDB never binds an unread CTE. They agree after PR 3a
because an unread CTE is ignored, as on DuckDB; the other 6 hit unrelated
refusals. So the generated grammar holds no case of a static-only CTE or
derived table actually joined to the request table, and PR 4 (computing one
at construction) has no measured value yet. Whether to build it, or first
widen the generator so the class exists, is the owner's call.

## Review record

Owner review, 2026-09-27, and what changed:

1. **Phase 2 must keep binding types apart from storage types.** Reproduced: the
   `NULL AS x` derived table binds `x || 'a'` as INTEGER, the materialized
   table as VARCHAR. Added the binding-schema/storage-schema split, one SQLNULL
   rule for both phases, and the materialization pin.
2. **`0 DIVERGE_VALUE` was not a sufficient gate.** Replaced by the acceptance
   gate above, enforced by the runner's exit status (it reported findings
   without failing).
3. **A construction refusal is not DuckDB's runtime trap.** Phase 2's trap is
   now stated as a refusal policy with a message, a verdict and a contract claim.
4. **"Recovers 150" overstated the evidence.** Now candidates, measured on a
   frozen snapshot after PR 3 and PR 4.
5. **Interface details.** Stage ownership, the shared query context, the global
   `many` join count and "once per row reaching the stage" are settled above.

The query context moved from PR 1 to PR 2 (2026-09-27): it has nothing to
share until a second binder exists, so PR 1 would only have added unused
scaffolding.

Rulings on the open questions: (1) staged pipeline approved, PR 1
behavior-preserving; (2) phase 2 approved under the conditions above; (3)
exclusions kept; (4) C1 default depth raised to 1,500
([C1 depth](../decisions/closed/c1-depth.md)).
