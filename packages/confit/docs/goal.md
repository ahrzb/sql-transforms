# The confit goal

**What this document is.** What confit is *for*, how much of it we intend to serve,
what we deliberately do not serve, and how each of those is measured. It sits above the
oracle spec (`packages/confit/docs/oracle/`) and the engine and testing specs that follow
it: the oracle spec defines *correct*, this document says *how much correct, over which
queries, and why the rest is out*.

**Non-circularity, the same rule the oracle spec runs on.** This document is the
authority. Code, tests, pins and gate floors are its **enforcement**, never its
definition. A goal is not "whatever the gates currently pass" — if it were, every
regression would redefine the goal the moment it landed. `Enforced-by:` names the thing
that makes a statement hold; `Verified-by:` names the test, pin, gate line or dated
measurement that would catch it not holding. A statement nothing checks says `Unverified`
and says so plainly.

**Facts are measured or read from a pin, never recalled.** Every number here was either
produced by running the thing on the date given, or read out of the file that pins it,
with the file and line named. Where a number in another document disagrees with what ran
today, both are printed.

**How an item is named.** Nothing is named by a number. Every item carries a **slug** —
a short kebab-case noun phrase naming its *subject*, so the name still reads true when
the ruling changes. Slugs are assigned once; renaming one is a tombstone line naming
both. The family is carried by the citation, not by the slug:

| kind | written as | example |
|---|---|---|
| goal | `goal: <slug>` | goal: two-outcome-contract |
| exclusion-ledger row | `exclusion: <slug>` | exclusion: whole-relation-shapes |
| KPI | `kpi: <slug>` | kpi: acceptance-rate |
| claim of fact | `claim: <slug>` | claim: corpus-match-today |
| ASK block | `ask: <slug>` | ask: acceptance-target |

Slugs are unique across every family and across the oracle spec, and the form above is
used at the definition and at every reference, so `grep -rn "<slug>"` finds an item and
everything that cites it.

**Two markers keep the normative half honest.** `[PROPOSED]` — a statement this document
would like, which nobody has ruled on; it holds a slug only so a ticket can cite it.
`[FACT]` — a measured statement of current state with no decision attached. Every
decision that belongs to the owner is an ASK block or carries `[PROPOSED]`; none of them
is written as settled. That includes every acceptance-rate target, every choice of which
query class is next, ratification of the exclusion ledger, any change to the KPI set, and
whether this document absorbs `packages/confit/docs/kpis.md`.

**Measurement environment**, so the dated numbers are reproducible: worktree at master
`2ba96e5`, DuckDB 1.5.5, release build via `uv run maturin develop --release`, Windows 11
/ 12 cores, all campaign runs at `--workers 8 --timeout 20`. Every number tagged
*2026-09-02* was produced there.

---

## 1. What confit is for

The repository has **two** goals, and confit is one half of one of them: ergonomic
SQL-to-transformer authoring, and fast inference. `sql_transform` owns authoring and fit;
confit owns the serving engine. This document is only about confit's half
(goal: engine-half-only).

**goal: serving-without-skew.** A feature transform written once, in SQL, produces the
same values at request time that it produced over the training table — bit for bit, not
approximately. Train/serve skew is the failure this engine exists to make impossible, and
"bit-exact" rather than "close" is the whole point: an engine that is 99% compatible does
not fail on 1% of queries, it silently corrupts a fraction of rows on queries it appears
to support.
*Enforced-by:* the two-outcome contract below, and the fit-side gates in
`packages/sql-transform` (KPI C1).
*Verified-by:* `packages/confit/README.md:11-22` (the contract and this argument);
`packages/confit/docs/kpis.md:29-41` (C1) and `:44-64` (C2).

**goal: two-outcome-contract.** For any SQL handed to `DuckDBInferFn`, exactly one of two
things happens: it serves bit-for-bit identical to the oracle, or it refuses at build with
a `ValueError` naming the construct. There is no third mode — nothing is approximated,
silently dropped, or widened at inference time. This is the load-bearing goal: everything
in section 2 follows from it.
*Enforced-by:* build-time refusal at `DuckDBInferFn(...)` construction throughout
`packages/confit/src/specializer/`.
*Verified-by:* `packages/confit/docs/kpis.md:90-98` (KPI C5, "no third mode", the corpus's
FAILED bucket pinned empty); `packages/confit/tests/test_corpus_replay.py:171-190` (three
outcomes, zero FAILs); `packages/confit/docs/properties.md:231-235` (P18);
`packages/confit/docs/oracle/` claim: oracle-identity for which DuckDB.

**goal: pack-time-only-work.** All general work happens once, at pack time: parse once,
bind once, compile once, freeze the statics into the code. Nothing general remains at call
time. The ceiling this doctrine tolerates is a single n-dispatch branch; anything that
would re-do general work per row is refused rather than served slowly.
*Enforced-by:* the specializer's partial-evaluation model; refusals for every construct
that would need per-row compilation (exclusion: per-row-general-work).
*Verified-by:* `packages/confit/docs/known-limitations.md:65-79` (section 1, "the
specialization bargain"); `packages/confit/docs/properties.md:246` (P20, statics frozen at
build); `packages/confit/docs/specs/2026-08-25-task-133-join-keys-design.md:520` ("an
engine whose whole doctrine is compile-once with no runtime ...").

**goal: request-latency-budget.** Serving cost is a single-digit-microsecond,
in-process, per-request number — the regime where a model server calls a feature
transform inside its own request, not the regime where a warehouse scans a batch.
DuckDB-per-call is three orders of magnitude off that and is not a competitor for it.
*Verified-by:* measured 2026-09-02 (claim: bench-is-stale): `spec` p50 at n=1 is
2,300-5,700 ns per call across the five `benchmarks/` scenarios, against 6.3-13.9 ms for
DuckDB-per-call on the same rows. **No target number is in force** — the budget is a
regime, not a bound anyone has ruled on. See ask: kpi-set-change.

**goal: engine-half-only.** Confit's scope stops at the engine: SQL plus frozen tables
plus declared UDF objects in, a native function out. Authoring sugar, window
marginalization, transformer fitting, and parity against sklearn belong to
`packages/sql-transform` and are not confit's to own or to claim
(claim: model-surface-split).
*Verified-by:* `README.md:12-13` (the two-package split);
`packages/confit/tests/test_tree_predict.py:9-11` ("Parity against sklearn is a separate
gate that lives in sql-transform").

**goal: growing-accepted-surface.** The quantity that is actually optimized is not parity
— parity is fixed at 100% by goal: two-outcome-contract — it is **acceptance**: how much
real SQL builds instead of refusing. Progress is queries moving from REFUSED to served,
never into a wrong answer. Section 2 is this goal restated with numbers.
*Enforced-by:* the ladder counts and gate floors named in section 2.
*Verified-by:* `packages/confit/docs/kpis.md:100-116` (D1, "Progress = moving queries from
REFUSED to MARGINALIZED (never to FAILED)").

---

## 2. Parity is a control; acceptance is the goal

The reframe this document exists to make explicit. Because the contract is
bit-exact-or-refuse-by-name (goal: two-outcome-contract), **parity on the accepted surface
is a control fixed at 100%**: a parity violation is a bug to fix, never a dial to trade.
Asking "how much oracle parity do we want" has exactly one answer, and it is not
interesting. The goal-shaped question is **which queries are accepted at all**, and how
fast that set grows.

Three verdicts carry the distinction, and the oracle spec defines them
(`packages/confit/docs/oracle/04-verdicts-agreement-abstention-refusal.md`):

| verdict | means | counts as |
|---|---|---|
| `AGREE` | ours == optimizer-off DuckDB == optimizer-on DuckDB | the only thing counted as coverage (claim: coverage-accounting) |
| `REFUSED` | confit refused at build; the case never entered the comparison | **not** a finding and **not** coverage — absorbed into a histogram (claim: refusal-absorb). This is the number this section is about |
| `UNSHIPPED` | the oracle's answer has a width we have not shipped, so *nothing was compared* | neither agreement nor finding; reported in its own section (claim: unshipped-verdict) |

`REFUSED` and `UNSHIPPED` are the two shapes of "not accepted", and they retire
differently: a refusal retires by a decision to serve the construct, an unshipped width by
the width landing.

### 2.1 The accepted surface, measured

**claim: corpus-match-today.** **[FACT]** Of the 678 statements mined from DuckDB's own
test suite, **547 replay bit-exact, 131 refuse cleanly, 0 FAIL** — measured 2026-09-02 by
`uv run pytest packages/confit/tests/test_corpus_replay.py -s`. Six documents quote
**550**, and the oracle spec's ladder records 550 as a genuine earlier reading
(`53 -> 395 -> 505 -> 511 -> 529 -> 546 -> 550`), so the ungated match count has moved
**down by three** since it was last written down and nothing noticed. Zero FAILs is the
gate and it holds; the match count is deliberately ungated.
*Verified-by:* the run above; `packages/confit/tests/test_corpus_replay.py:171-190` (the
zero-FAIL gate); `packages/confit/docs/oracle/` claim: zero-fails-gate and its correction,
which names all six stale sites. This measurement is fresh evidence for the oracle spec's
open ask: match-count-ratchet — a ladder nothing ratchets has now demonstrably slipped —
and this document does **not** fork a second question about it.

**claim: dialect-floors-today.** **[FACT]** The dialect frontend's L2 gate (parse then
print is invisible to the oracle) stands at **288/678 match, 390 clean-unsupported, 0
FAIL** — measured 2026-09-02, exactly on its floor of 288
(`packages/confit/tests/test_dialect_corpus_gate.py:37`). The L3 cross-engine gate's Spark
floor is **260** (`test_dialect_cross_engine_gate.py:50`) and **could not be measured
here**: `pyspark` is not installed in this environment, and the fixture fails loudly by
design rather than skipping. Both floors carry the ratchet — "raise it when the surface
grows, never lower it".
*Verified-by:* the run above; the two floor constants named; `packages/confit/docs/oracle/`
claim: dialect-gate-oracle.

**claim: campaign-verdicts-today.** **[FACT]** Over seeds 0-1999 of the generated grammar,
measured 2026-09-02 and reproduced twice (`python -m fuzz.runner --seed 0 --n 2000`, and
independently through `fuzz.oracle.run_case_json`, identical counts both times):

| verdict | count | share |
|---|---|---|
| `AGREE` | 1013 | 50.7% |
| `REFUSED` | 944 | 47.2% |
| `AGREE_TRAP` | 21 | 1.1% |
| `UNSHIPPED` | 14 | 0.7% |
| `DIVERGE_OPT` | 7 | 0.35% |
| `DIVERGE_VALUE` | 1 | 0.05% |

**Acceptance** — cases that built rather than refused — is **1056/2000 = 52.8%**. All 14
`UNSHIPPED` are class `decimals` (exclusion: unshipped-decimal-arithmetic). All 7
`DIVERGE_OPT` are the accepted optimizer cost (exclusion: optimizer-on-answers). The one
`DIVERGE_VALUE`, seed 1804, is a **live parity finding**: `CAST(pow(-0.25e0, 0.1e0) AS
VARCHAR)` inside `struct_pack` gives `'nan'` on one side and `'-nan'` on the other — a NaN
*sign* reaching a string, which the canonical `repr` comparison elsewhere cannot see
(claim: blind-spots' NaN row). It is recorded here, not fixed here; per the engine-bug
process it wants an xfail-strict pin and a ticket.

**Campaign-validity caveat, and it bounds every number above.** These rates are over the
**generated grammar**, which is not query space. The oracle spec records the caveats
directly: the campaign fuzzer is a manual CLI and *not* a standing gate — only the regexp
fuzzer is (claim: regexp-fuzz-gate's correction); there is no coverage denominator that
means anything yet (claim: coverage-denominator, `[PROPOSED]`); a rising abstention rate
would read as generator drift and is not reported (claim: abstention-rate, `[PROPOSED]`);
and the mined corpus's expected rows are an optimizer-**on** recording with no provenance,
so the 678-statement ladder is measured against a different reading of DuckDB than
claim: oracle-identity names (claim: zero-fails-gate's second correction). "52.8% of a
grammar" is a real measurement of a synthetic population, and nothing here converts it
into a statement about the SQL people write.

### 2.2 The authoring-side ladder

`packages/confit/docs/kpis.md` D1 pins the sql-transform admission ladder: **11
marginalized + 11 refused of 22 mined**, and **39 marginalized / 17 refused / 5
schema-mode** curated. Verified fresh 2026-09-02 —
`_corpus_test.py::test_progression_totals` passes, so the pins in kpis.md and the pins in
the test agree. That ladder is `sql_transform`'s, not confit's (goal: engine-half-only);
it is cited here because acceptance growth is measured on both sides of the boundary and
only one side has a pin test that fails when the number moves.

> ### ask: acceptance-target — is there an acceptance-rate target, and what is it?
>
> Acceptance is 52.8% over the generated grammar and 547/678 over the mined corpus
> (claim: campaign-verdicts-today, claim: corpus-match-today). Neither number has a target
> and neither has a ratchet. Three shapes, and the choice is yours:
>
> **(a) No target, dated reporting only.** Cheapest, and honest about the denominator
> problem: a grammar rate is not a query-space rate, so a target on it optimizes the
> generator as readily as the engine. Cost: nothing notices a slip — the corpus ladder
> just lost three cases silently, which is claim: corpus-match-today.
>
> **(b) Ratchet, no target.** "Never decreases", the rule the dialect gates already run
> (claim: dialect-floors-today). Cost, and it is real: a deliberate scope *reduction*
> becomes a gate failure. The oracle spec's ask: match-count-ratchet is the same question
> for the mined corpus and should be answered with this one, not separately.
>
> **(c) A named target per surface**, e.g. "70% acceptance over the campaign grammar by
> the time decimal arithmetic ships". Cost: it needs the denominator work
> (claim: coverage-denominator) before the number means anything.
>
> My read: (b) for the corpus and dialect ladders, which have stable denominators, and (a)
> for the campaign rate until a denominator exists. Not applied — this is your call.
>
> *Binds:* goal: growing-accepted-surface, kpi: acceptance-rate, kpi: ladder-ratchet, and
> the oracle spec's ask: match-count-ratchet.

> ### ask: next-query-classes — which classes are next, and in what order?
>
> The candidates, each with what it would unlock and what it costs. None of these is
> chosen; the list is evidence for a choice, not a plan.
>
> | candidate | unlocks | cost / blocker |
> |---|---|---|
> | decimal arithmetic (m-8 lattice phase 5) | retires exclusion: unshipped-decimal-arithmetic entirely; empties the only `UNSHIPPED` bucket (14/2000 today) | an exact decimal lane through the expression tree, not just the join lane that already ships |
> | HUGEINT + the unsigned family (the i128 lane) | retires half of exclusion: wide-integer-lanes; the cranelift dependency was verified GO 2026-08-15 | i128 arithmetic and its overflow traps across both backends |
> | narrow-lane overflow traps (m-8 phase 3) | closes the one place a narrow lane serves an i64 value on the row path where `infer_arrow` refuses | trap threshold per declared width |
> | nested / struct-valued outputs | retires the whole-struct half of exclusion: non-scalar-values | output schema work at the Arrow boundary |
> | `USING` / `NATURAL` self-joins under `shape='many'` | closes a named follow-up inside exclusion: multiplicity-by-default | small; it is a named rejection, not a model gap |
> | lifting the one-join-per-query restriction under `'many'` | multi-join serving | multiplicity composition, the hardest of these |
>
> The campaign says which of these the *generator* reaches, not which your users need —
> `WITH` (104 refusals of 944), `comparison on BOOLEAN` (86) and `QUALIFY` (36) top the
> refusal histogram, and the first and third are permanent by
> exclusion: whole-relation-shapes. That is a fact about the grammar, not a demand signal.
>
> *Binds:* goal: growing-accepted-surface, exclusion: unshipped-decimal-arithmetic,
> exclusion: wide-integer-lanes, exclusion: non-scalar-values,
> exclusion: multiplicity-by-default.

---

## 3. What we exclude for now

The ledger. Each row: **what** is excluded, **why**, its **retirement trigger**, and **how
it rings** when the trigger fires — or an honest flag that nothing rings.

Three grounds classify every exclusion, and they are already decided:
**specialization-inherent** (the engine model cannot express it — permanent),
**scope-by-product-decision** (it could be served and we chose not to — reversible by a
decision), **resource** (it would cost more than a serving engine may spend per row — a
judgement with a number attached). See `packages/confit/docs/oracle/`
claim: refusal-grounds.

**Not in this ledger, deliberately:** the rows of `known-limitations.md` section 5 that
*serve* with a consciously different surface (duplicate-column rename, approximate error
texts, schema-qualifier resolution, the platform-libm NaN bit pattern). Those are
divergences, not exclusions, and they belong to the oracle spec's divergence ledger
(`packages/confit/docs/oracle/07-the-divergence-ledger.md`). Two entries below overlap that
ledger because the *decision* is an exclusion even though the *symptom* is a divergence;
each says so.

**exclusion: whole-relation-shapes.** `GROUP BY`/`HAVING`/aggregates, `ORDER BY`,
`LIMIT`/`OFFSET`/`FETCH`/`TOP`, `DISTINCT`, CTEs, `UNION`/`INTERSECT`/`EXCEPT`, subqueries,
multiple statements, table functions in `FROM`, `rowid`, `FULL OUTER JOIN`.
*Why:* scope-by-product-decision. Their output shape is not one-row-in / 0..N-out, so they
are not row-at-a-time feature transforms. Measured share: `WITH` alone is 104 of 944
campaign refusals, `QUALIFY` 36, `DISTINCT` 30, `ORDER BY` 23.
*Retirement trigger:* none intended. The carve-out already exists and is not a retirement:
a **static-tables-only** query is evaluated once by DuckDB at build and frozen, so
aggregation and `ORDER BY` serve there — with row limits refused, because which rows
survive a limit is not a function of the query (measured: four different answers across
twelve connections).
*Rings:* nothing rings — there is no trigger. Lifting one breaks
`packages/confit/tests/test_known_limitations.py`, which is the executable twin of every
row here.
*Verified-by:* `packages/confit/docs/known-limitations.md:95-120`.

**exclusion: per-row-general-work.** Non-constant regex patterns, replacement strings,
regex options and extract-group indexes; anything that would compile or bind per row.
*Why:* specialization-inherent, and the direct negation of goal: pack-time-only-work.
DuckDB compiles regexes per row; we compile at prepare.
*Retirement trigger:* a decision to abandon compile-once for this construct. Effectively
permanent — retiring it contradicts a goal, so it would be a goal change first.
*Rings:* nothing rings.
*Verified-by:* `packages/confit/docs/known-limitations.md:74-75`.

**exclusion: unshipped-decimal-arithmetic.** Expressions over DECIMAL — arithmetic, `CAST`
to anything but `DOUBLE`, and `COALESCE`/`CASE`/`greatest` unifying a decimal with a
non-identical type. DECIMAL *literals* are typed f64 rather than DuckDB's `DECIMAL(p,s)`,
with the visible consequence that `CAST(-2.5 AS BIGINT)` on a bare literal rounds half-to-
even here and half-away-from-zero there. Decimal *statics* serve exactly, in an i128 lane,
and are **not** excluded.
*Why:* scope-by-product-decision, pending the lattice phase. Each of these used to serve a
silently wrong double, which is why they refuse now rather than approximate.
*Retirement trigger:* **decimal arithmetic ships** (m-8 lattice phase 5).
*Rings:* **yes, loudly, and this is the one exclusion with a real bell.** The campaign
classifies these cases `UNSHIPPED` instead of comparing them, `fuzz.oracle._type_delta`
carries exactly one arm (decimal-against-float64) which is deleted when the feature lands,
and the runner prints the bucket in a section of its own — an empty section means either
the feature shipped or the grammar stopped reaching it, and both are worth seeing.
Measured 2026-09-02: 14 of 2000 seeds, all class `decimals`.
*Verified-by:* `packages/confit/docs/known-limitations.md:138-174`;
`packages/confit/fuzz/oracle.py:126-143` (the note), `:498-505` (the arm),
`packages/confit/fuzz/runner.py:168-176` (the report section);
`packages/confit/tests/test_fuzz_smoke.py::test_an_unshipped_lane_is_classified_and_never_value_compared`.

**exclusion: wide-integer-lanes.** HUGEINT and the whole unsigned family refuse by name
rather than collapse to i64; `float32` base tables refuse; `float32` and unsigned *static*
columns refuse rather than widen. Narrow lanes type in DuckDB's lattice but compute in i64,
so until the trap phase lands an overflowing narrow lane serves the i64 value on the row
path and refuses at the `infer_arrow` boundary.
*Why:* specialization-inherent for the lane widths (the engine computes in i64/f64/str/bool),
scope-by-product-decision for the refusal-instead-of-widen rule. The catalogue used to
widen `float32` and the unsigned widths and **both diverged silently**, which is the
measurement that produced the rule.
*Retirement trigger:* the **i128 lane** ships (cranelift dependency verified GO
2026-08-15) for HUGEINT/unsigned; **m-8 phase 3** for the narrow-lane traps.
*Rings:* partially — `packages/confit/tests/test_integer_widths.py` pins the width
catalogue, so a shipped width forces an edit there. Nothing rings for the *unsigned*
family specifically.
*Verified-by:* `packages/confit/docs/known-limitations.md:122-136`, `:175-187`.

**exclusion: non-scalar-values.** List-typed columns when referenced (unreferenced they
cost nothing), the struct as a whole value (`SELECT a`), bracket field access (`a['i']`),
struct fields whose own types are non-scalar, the list-valued regex functions
(`regexp_extract_all`, `regexp_split_to_array`, the STRUCT form of `regexp_extract`),
`decimal256` statics, decimal row columns, and the BLOB overload DuckDB picks for a bare
`NULL` `repeat` string. Structs of scalars, deep field paths and struct-star **do** serve.
*Why:* scope-by-product-decision — the row-schema vocabulary is scalar, and a non-scalar
output has no place in a dict row.
*Retirement trigger:* nested output support for the struct half; the m-8 Blob phase for
BLOB; `decimal256` is upstream-blocked (DuckDB itself refuses it at arrow register).
*Rings:* nothing rings.
*Verified-by:* `packages/confit/docs/known-limitations.md:152-166`, `:207`.

**exclusion: parse-divergence-guards.** The `^` operator (it *is* pow in DuckDB, but
sqlparser's precedence differs, so mapping it computes a different tree silently), prefix
`~`, `#`, `NOT GLOB`, and the regex reject list: the RE2-vs-rust-regex differential
battery's classes plus the twelve the fuzzer found.
*Why:* specialization-inherent — these are the constructs where a served answer would be
*silently* different, which goal: two-outcome-contract forbids outright.
*Retirement trigger:* the dialect frontend's own parser replacing sqlparser for the
precedence cases; a matching regex engine for the RE2 cases. Neither is scheduled.
*Rings:* **yes** — the standing regexp differential fuzzer runs in the normal gate
(N=250, fixed seed), and a new divergence fails with its reproducing seed. It re-swept to
zero divergences over 40k cases across 8 seeds.
*Verified-by:* `packages/confit/docs/known-limitations.md:197-200`;
`packages/confit/tests/test_duckdb_regexp_fuzz.py:13-20, :35-37`;
`packages/confit/docs/oracle/` claim: regexp-fuzz-gate.

**exclusion: resource-ceilings.** Pad/repeat counts past a 1 GiB string-builder budget (a
literal count refuses at build; a data-driven count traps at runtime), the regex
program-size guard, and the Arrow batch ceiling.
*Why:* **resource** — no gigabyte allocations in a serving engine, by decision. This is
the ground whose whole point is that the accepted cost has a number attached.
*Retirement trigger:* a decision to raise the number, which is a review, not a feature.
*Rings:* the ceilings are pinned as executable divergences in
`packages/confit/tests/known_divergences/test_string_budget.py` and `test_arrow_boundary.py`
— they ring on a *change*, not on a retirement.
*Verified-by:* `packages/confit/docs/known-limitations.md:205`;
`packages/confit/tests/known_divergences/test_arrow_boundary.py:34-36`;
`packages/confit/docs/oracle/` divergence: string-builder-budget,
divergence: arrow-batch-ceiling.

**exclusion: optimizer-on-answers.** We do not reproduce DuckDB's 33 plan-rewrite passes.
The user-visible cost: a trapping subexpression the optimizer would delete, we still
evaluate — so a query that returns a value in the reader's own DuckDB session can raise
here. Overlaps the divergence ledger; the *decision* is an exclusion, the *symptom* is
divergence: trap-elision.
*Why:* scope-by-product-decision, on a measured ground. The optimizer-on reading is not a
function of the query — `statistics_propagation` answers from a column's stored null
statistic, so the same query over the same rows differs by the table's insert history. A
target you cannot compute from the query is not a target, and confit compiles against a
schema and never sees a table.
*Retirement trigger:* none intended. Decided 2026-08-17.
*Rings:* **yes** — the campaign reads DuckDB twice per case and labels these `DIVERGE_OPT`,
so the class is counted rather than absorbed. Measured 2026-09-02: 7 of 2000 seeds; a
4000-seed campaign put it at 8 seeds in 28 findings.
*Verified-by:* `packages/confit/docs/known-limitations.md:20-39`, `:231-257`;
`packages/confit/docs/oracle/` claim: oracle-identity, claim: optimizer-bracket.

**exclusion: statistics-dependent-kernels.** Behaviors that depend on column *statistics*
— ILIKE's NUL handling selects a different kernel depending on sibling rows — are excluded
from the corpus by name, and the engine takes the ASCII-kernel (NUL-transparent) behavior.
*Why:* specialization-inherent. A row-at-a-time engine cannot reproduce statistics-
dependent semantics even in principle.
*Retirement trigger:* none. This one is permanent by construction.
*Rings:* nothing rings, by design — the exclusion is a named source list, and a named list
does not fire.
*Verified-by:* `packages/confit/docs/known-limitations.md:225-230`;
`packages/confit/tests/test_corpus_replay.py:150-151` (`_KNOWN_DIVERGENT_SOURCES`);
`packages/confit/docs/oracle/` claim: statistics-dependent-exclusion.

**exclusion: multiplicity-by-default.** Duplicate-key joins, cross joins, and
inequality/constant `ON` joins build **only** under `shape='many'`, one join per query;
`shape='map'` rejects anything that can drop a row; `USING`/`NATURAL` self-joins refuse
under every shape.
*Why:* scope-by-product-decision. Multiplicity is never allowed to sneak into a serving
path by default — `map` is a build-time *proof* of exactly-one, not a runtime check.
Measured share: `shape='map': a WHERE clause can drop ...` is 42 of 944 campaign refusals.
*Retirement trigger:* the one-join-per-query restriction lifting, and `USING`/`NATURAL`
self-joins landing — both named follow-ups, neither scheduled.
*Rings:* nothing rings for the retirement; the shape contract itself is pinned in
`packages/confit/tests/test_shape_contract.py`.
*Verified-by:* `packages/confit/docs/known-limitations.md:76-93`.

> ### ask: exclusion-ratification — does this ledger become the exclusion list?
>
> Ten rows above, assembled from `known-limitations.md`, the engine's refusal sites and
> the campaign's refusal histogram. Ratifying them means three things bind: the **ground**
> assigned to each (specialization-inherent vs scope-by-product-decision vs resource, which
> is what makes "we refuse" auditable), the **retirement trigger** named, and the fact that
> **six of the ten ring nothing** — nothing in the gate notices when their trigger fires.
>
> Two specific pieces I would like ruled on rather than assumed:
>
> **(1) Are the six silent rows acceptable?** exclusion: whole-relation-shapes,
> per-row-general-work, non-scalar-values, statistics-dependent-kernels and
> multiplicity-by-default ring nothing, and exclusion: wide-integer-lanes rings only
> partially. That is fine for the ones whose trigger is "never" — it is a live gap for
> exclusion: non-scalar-values and exclusion: multiplicity-by-default, whose triggers are
> real scheduled-ish work. The pattern that *does* work is
> exclusion: unshipped-decimal-arithmetic's: a classified verdict, a single code arm
> deleted on landing, and a report section that empties.
>
> **(2) Is the enumeration complete?** It is assembled, not derived — the engine has 281
> `unsup(...)` call sites across nine source files (measured 2026-09-02:
> `specializer/frontend.rs` 144, `dialect/duckdb.rs` 84, `specializer/retrans.rs` 22,
> `dialect/plan.rs` 13, and five files with fewer than 10), and no mechanism maps a call
> site to a ledger row. A refusal that exists in code and in no document is exactly the
> bookkeeping bug `known-limitations.md:284` asks readers to file.
>
> *Binds:* every `exclusion:` slug in this section, and
> `packages/confit/docs/known-limitations.md` as their source.

---

## 4. What we cover that DuckDB does not

The model surface. DuckDB is the oracle for SQL; it has no opinion at all about a fitted
sklearn transformer or a gradient-boosted tree, so on this surface there is no differential
oracle and a different reference takes its place.

**claim: model-surface-split.** **[FACT]** The surface is owned by two packages and the
line is drawn at the artifact, not at the algorithm. **Confit owns**: the `udfs=` protocol
(a declared object with `name` / `takes` / `returns` / optional `instances` and a scalar
`__call__`), the extern call machinery, the STRUCT-valued return and its field access, and
the native tree kernel — a UDF exposing `tree_tables()` is scored by native code from a
pair of Arrow tables plus a grid, with no sklearn import anywhere in the package.
**sql-transform owns**: fitting, clone-per-group semantics, the packing of an sklearn
estimator into those tables, and **parity against sklearn**.
*Verified-by:* `packages/confit/tests/test_tree_predict.py:1-17` — "Nothing here imports
sklearn ... DuckDB has no native tree scoring, so there is no differential oracle here ...
Parity against sklearn is a separate gate that lives in sql-transform";
`packages/confit/tests/test_udfs.py:1-10` ("this package's contract is the protocol, not
the class").

**claim: udf-parity-is-still-the-oracle.** Where a UDF *can* be registered with DuckDB
(`con.create_function`), the contract does not weaken: confit serves bit-for-bit identical
to DuckDB **with the same UDFs registered**, or refuses. The UDF surface is the ordinary
contract with one parameter, not an exemption from it — which is why a UDF named after a
builtin is refused rather than resolved: DuckDB lets a registered function shadow its
builtin and we do not, so serving it would be two engines answering one SQL differently.
*Enforced-by:* `packages/confit/tests/test_udfs.py::udf_check` (`:240`), the parameterized
form of the contract.
*Verified-by:* `packages/confit/docs/kpis.md:44-56` (C2 names `test_udfs.py` among its
enforcing suites); `test_udfs.py::test_a_udf_may_not_take_a_builtin_name`.

**claim: sklearn-is-the-reference.** On the surface DuckDB cannot run, an independent
sklearn reference plays the role optimizer-off DuckDB plays for SQL — and it is a
**reference**, not the oracle, which is why it comes with a named bound instead of bit
equality everywhere. The bound differs by family and both halves are in force today: tree
scoring is **bit-exact** against `sklearn`'s own `predict`, while fitted transformer
columns gate against a clone-per-group reference at **rtol 1e-9**. The campaign's sklearn
metamorphic leg uses the same `1e-9`.
*Enforced-by:* `packages/sql-transform/sql_transform/_trees_test.py::test_matches_sklearn_bit_exactly`;
`packages/sql-transform/sql_transform/_transformers_test.py::_reference` (`:34`) with
`rtol=1e-9` at `:109, :127`; `fuzz.oracle._extra_legs`' sklearn leg.
*Verified-by:* `packages/confit/docs/kpis.md:76-87` (C4, the control this is);
`packages/confit/docs/oracle/` claim: metamorphic-self-legs (the `1e-9` leg, itself
recorded as having **no test of its own** — `Unverified`).
*Note, and it belongs to the owner rather than to this document:* C4's extension for
native transform families ("bit-exact for scaler/tree tiers, within the declared
per-family ulp bound for matvec tiers") is written in kpis.md against work that has not
landed. It is recorded here as a pointer, not adopted.

---

## 5. Measurement and KPIs

### 5.1 The standing law

Quoted from `packages/confit/docs/kpis.md`, and nothing proposed below may weaken it:

> Never trade a control for a drive gain. If a bar seems in the way, the move is a
> written, named tolerance — or a refusal.

and its companion, `:17-20`: "**Loosening a control is a design decision, never a fix.**
The only legitimate way a control moves: explicitly, in a spec/draft, with the new bound
named."

Every `[PROPOSED]` KPI in 5.3 is a **drive** or a **zero-control**, and none of them
touches C1-C5.

### 5.2 What is measured today

**claim: kpi-pointers-resolve.** **[FACT]** Every enforcement pointer under the five
control KPIs in `packages/confit/docs/kpis.md` resolves on master `2ba96e5`, checked
2026-09-02: C1 `_projection_test.py::gate` (`:34`) and its `MARGINALIZE_FUZZ_N` seeded
differential (`:374-376`, seed 20260729); C2 the `test_duckdb_*.py` wave suites,
`test_params_joins.py`, `test_udfs.py::udf_check` (`:240`), `known-limitations.md`, and
`src/specializer/exec/tests.rs`; C3 `_serving_test.py::serve_gate` (`:29`); C4
`_transformers_test.py::_reference` (`:34`); C5 `_corpus_test.py` and its three-outcome
FAILED-must-be-empty rule. Two honest notes: C1's text says "1,500-2,000-case runs at each
widening loop", but the **default** `MARGINALIZE_FUZZ_N` is **25** — the deep run is
opt-in, so the gate as it runs is two orders of magnitude shallower than the KPI reads.
And the D1 pins in kpis.md (`11/22`; `39/17/5`) match
`_corpus_test.py::test_progression_totals` exactly, so D1 is fresh.
*Verified-by:* the file and line reads above, plus a passing run of
`test_progression_totals` on 2026-09-02.

**claim: bench-is-stale.** **[FACT]** D2's serving-latency table is dated **2026-08-04**
at master `a6fa318`, which is **389 commits** behind `2ba96e5`. Worse than age: its
headline claim — "~1.5-1.7x faster than a handwritten Python microservice twin" — compares
against a harness row that **no longer exists**. kpis.md itself records that the typed-model
`python` and `spec_dict` rows retired with the pydantic surface. Re-measured 2026-09-02
(`uv run python -m benchmarks.bench_serving`, parity gate green on all five scenarios), the
comparable pair is `spec` against the surviving `python_dict` row, p50 ns at n=1:

| scenario | spec | python_dict | ratio | duckdb per call |
|---|---|---|---|---|
| titanic | 3,500 | 2,300 | 1.52x slower | 7,676,800 |
| house_prices | 5,300 | 4,400 | 1.20x slower | 13,895,800 |
| fraud_txn | 5,700 | 4,600 | 1.24x slower | 13,419,700 |
| store_sales | 5,600 | 4,000 | 1.40x slower | 12,960,800 |
| feature_bundle | 2,300 | 2,300 | 1.00x | 6,263,600 |

The three-orders-of-magnitude gap to DuckDB-per-call (goal: request-latency-budget) is
intact. The sign of the Python comparison is not: the recorded claim says faster, today's
run says slower on four of five scenarios and level on the fifth. **This document does not
call that a regression** — the baseline row changed identity (plain dicts out, where the
old row returned pydantic models), the machine is not the 2026-08-04 machine, and kpis.md
warns absolute numbers drift with load. It is a sign flip in a claim nobody re-ran for a
month. See ask: bench-baseline-flip.

**claim: refusal-prefix-share.** **[FACT]** Refusal *quality* is measurable today and has
never been measured. Over seeds 0-1999, 2026-09-02: of 944 refusals, **837 (88.7%)** carry
one of the three documented prefixes (`unsupported:` / `parse error:` / `bind error:`);
**107 (11.3%)** carry none. **688 (72.9%)** fall in the corpus gate's narrower `_CLEAN`
set, which excludes `bind error:` entirely. The undocumented 107 are three families, and
**two of them are not named in the oracle spec's existing correction**:
`shape='map': a WHERE clause can drop ...` (42) and its static-only twin (1);
`udf '<name>': a width-1 list return ...` (63); `static data mismatch: @0: duplicate ...`
(1). The oracle spec's claim: refusal-message-prefixes already records that the prefix set
is not exhaustive and routes the fix through ticket: clean-prefix-reconcile; these two
families are new members for that ticket, and this document does **not** fork a second
question about them.
*Verified-by:* the measurement above; `packages/confit/tests/test_corpus_replay.py:36`
(`_CLEAN`); `packages/confit/docs/known-limitations.md:274-285` (the documented three);
`packages/confit/docs/oracle/` claim: refusal-message-prefixes and its correction.

### 5.3 Proposed KPI refinements

Six candidates. Each names the measurement that would back it and what it costs. **None is
adopted** — the KPI set is `packages/confit/docs/kpis.md`'s and changing it is
ask: kpi-set-change.

**kpi: acceptance-rate.** **[PROPOSED]** A **drive**: the fraction of generated-grammar
cases that build rather than refuse, reported per campaign with its seed range.
*Measurement that backs it:* exists and runs today — 52.8% over seeds 0-1999
(claim: campaign-verdicts-today), reproducible and deterministic.
*Cost:* the denominator is a grammar, not query space, so the number can be moved by
editing the generator. Adopting it as a *drive* without claim: coverage-denominator's
triple-axis work invites optimizing the wrong thing. Adopting it as a *reported statistic*
costs nothing.

**kpi: findings-per-campaign.** **[PROPOSED]** An explicit **zero-control**: findings per
N seeds, excluding `DIVERGE_OPT` (which is exclusion: optimizer-on-answers' accepted cost,
not a defect). Today: **1 per 2000** — the seed-1804 NaN-sign case.
*Measurement that backs it:* `findings.jsonl` already carries exactly this, per kind and
per class.
*Cost, and it is the reason to think:* the campaign fuzzer is **not** a standing gate —
`packages/confit/fuzz/` is a manual CLI, `testpaths = ["tests"]` does not collect it, and
`test_fuzz_smoke.py`'s own docstring says "'no findings over N seeds' **cannot** be the CI
invariant". Making findings-per-N a control would mean either wiring the campaign into CI
(minutes per run, and the value of a fuzzer is finding live bugs, which a zero-control
punishes) or declaring a control nothing enforces. The honest middle is a control on the
**release** cadence, not the commit cadence.

**kpi: unshipped-burndown.** **[PROPOSED]** A **drive**: the `UNSHIPPED` bucket's size and
its class list, driven to empty. Today **14/2000, one class (`decimals`)**.
*Measurement that backs it:* the runner already prints the bucket in its own section, and
`_type_delta` carries one arm per unshipped feature — so the bucket is exactly the ledger's
open widths, already machine-readable.
*Cost:* near zero; this is the one candidate whose machinery is already built and whose
retirement bell already rings (exclusion: unshipped-decimal-arithmetic). Note the empty
state is ambiguous — the runner's own comment says an empty section means either the
feature shipped **or the grammar stopped reaching it**, so burn-down to zero needs the
generator checked, not just the number.

**kpi: named-refusal-share.** **[PROPOSED]** A **control** at 100%: every refusal carries a
documented, actionable prefix naming the construct. Today **88.7%**
(claim: refusal-prefix-share).
*Measurement that backs it:* the measurement in 5.2, which is a 40-line script over the
campaign's own verdicts.
*Cost:* a control that starts at 88.7% is, by kpis.md's own definition, "not a control, it
is an unacknowledged trade-off" — so adopting it means **first** closing the 107, which is
ticket: clean-prefix-reconcile's work, and only then declaring the bar. Adopting it as a
drive first, then promoting it, is the sequence that respects the standing law. This is the
candidate I would rank first: it is the only one that measures whether
goal: two-outcome-contract's *second* outcome is actually usable, and nothing measures that
today.

**kpi: ladder-ratchet.** **[PROPOSED]** The existing D1 drive, extended: the corpus match
count (547/678), the dialect L2 floor (288) and L3 Spark floor (260), and the sql-transform
ladder (11/22; 39/17/5), each with a never-decreases rule and a single dated home.
*Measurement that backs it:* all four already exist; two already ratchet
(claim: dialect-floors-today), two do not.
*Cost:* a deliberate scope reduction becomes a gate failure. That cost is not theoretical —
the corpus count has already slipped 550 to 547 unnoticed (claim: corpus-match-today),
which is the argument *for*; and a scope reduction is sometimes right, which is the argument
*against*. Same question as the oracle spec's ask: match-count-ratchet; answer them
together.

**kpi: bench-refresh-cadence.** **[PROPOSED]** D2 carries a **staleness bound**: the
serving bench is re-run and re-recorded at a named cadence (per release, or per N commits),
and a number older than the bound is marked stale rather than quoted.
*Measurement that backs it:* the bench runs in a few minutes today with a green parity gate
(claim: bench-is-stale), so the cadence is affordable.
*Cost:* absolute numbers drift with machine load, so a cadence produces noise unless what is
recorded is the **ratio** to a baseline row in the same run — which is what kpis.md already
does for the transformer path ("the honest cross-run metric is the ratio of a 2-field query
to a 1-field one") and does *not* do for the pure-SQL table. Adopting this means picking
which ratio is the metric.

> ### ask: kpi-set-change — adopt any of the six, and as what?
>
> The KPI set is five controls and two drives, and changing it is yours alone. My ranking,
> with the reason, not as a recommendation to be rubber-stamped:
>
> 1. **kpi: named-refusal-share, as a drive now and a control later.** Nothing today
>    measures whether a refusal is usable, and a refusal that does not name its construct
>    fails goal: two-outcome-contract's promise as surely as a wrong value does. Adopting it
>    as a control at 88.7% would violate the standing law on its first day.
> 2. **kpi: unshipped-burndown, as a drive.** Its machinery already exists and already
>    rings; adopting it costs a line in kpis.md.
> 3. **kpi: ladder-ratchet**, decided together with the oracle spec's
>    ask: match-count-ratchet, because they are one question.
> 4. **kpi: bench-refresh-cadence**, once ask: bench-baseline-flip is settled — a cadence on
>    a metric whose baseline changed identity would just re-record the confusion.
> 5. **kpi: acceptance-rate**, as a *reported statistic* rather than a KPI, until a
>    denominator exists (claim: coverage-denominator).
> 6. **kpi: findings-per-campaign** last, and only at release cadence — a zero-control on a
>    fuzzer punishes the fuzzer for working.
>
> Adopting none is a legitimate answer and leaves kpis.md exactly as it is.
>
> *Binds:* all six `kpi:` slugs, and `packages/confit/docs/kpis.md`'s two-kind structure.

> ### ask: bench-baseline-flip — regression, or a baseline that changed identity?
>
> D2 records the engine as 1.5-1.7x faster than the handwritten Python twin. Today it is
> 1.0-1.5x **slower** than the twin row that still exists (claim: bench-is-stale), uniformly
> across five scenarios. Three readings, and one of them needs work to rule out:
>
> **(a) Baseline changed identity.** The old `python` row returned pydantic models; the
> surviving `python_dict` row returns plain dicts, which is cheaper. If that accounts for
> the whole delta, D2's prose is simply obsolete and the fix is an edit.
>
> **(b) Real regression.** 389 commits landed since the measurement, including the Arrow
> schema API, the lane/slot seams and join-key work. Ruling this out costs one bisect
> against `a6fa318` on the same machine.
>
> **(c) Machine noise.** Single run, laptop, other agents active. Cheap to rule out: three
> runs.
>
> I did not bisect — the fence for this task is one document. What I can say is that (c)
> alone is unlikely to produce a uniform sign flip across five scenarios, and that the
> answer changes what goal: request-latency-budget is worth as a claim.
>
> *Binds:* goal: request-latency-budget, kpi: bench-refresh-cadence,
> `packages/confit/docs/kpis.md:118-199` (D2).

---

## 6. Where this document sits

Above the specs, below the owner. The intended shape of the set:

| document | answers |
|---|---|
| **this document** (`docs/goal.md`) | what confit is for, how much of it we intend to serve, what we exclude, how it is measured |
| `docs/oracle/` (merged) | what *correct* means — the oracle's identity, the verdict taxonomy, the comparison contract, pins, the divergence ledger |
| the engine spec (upcoming) | how the engine achieves it — lanes, slots, the specializer, the backends |
| the testing spec (upcoming) | how it is checked — gates, corpora, campaigns, what each suite is for |

The direction of citation runs downward: this document may cite an oracle-spec claim as
evidence for a goal, and the oracle spec does not cite goals. Where the two overlap the
oracle spec wins on *correctness* questions and this document wins on *scope* questions —
which is why section 3 does not re-litigate the divergence ledger and section 2 does not
redefine a verdict.

> ### ask: kpis-absorb-or-defer — does this document absorb kpis.md, or defer to it?
>
> `packages/confit/docs/kpis.md` currently holds both halves of a KPI: the *definition*
> (what C2 means, why a control is never traded) and the *reading* (the D2 latency tables,
> the D1 pins). This document needs the first and keeps re-quoting it.
>
> **(a) Absorb.** Move the five controls, the two drives and the standing law here; kpis.md
> becomes a tombstone pointing at this file. Pro: one home, and the definitions sit next to
> the goals they serve. Con: kpis.md is cited from the oracle spec (claim: fit-serving-oracle
> cites `kpis.md:31-34` and `:76-87` by line), from `properties.md`, from drafts and from
> merged PRs — absorbing it invalidates line-anchored citations across the tree.
>
> **(b) Defer, and split by kind.** kpis.md keeps the KPI *definitions* and the standing
> law; the dated *readings* move to whatever runs them (the bench prints its own table; the
> ladder pins live in the test). This document cites kpis.md and never restates a bar. Pro:
> no citation breakage, and it fixes the real defect — a document holding month-old
> measurements as prose is how claim: bench-is-stale happened. Con: two files.
>
> **(c) Defer wholly**, status quo plus a pointer. Cheapest, changes nothing.
>
> My read is **(b)**: the staleness this document measured in D2 and in the corpus count is
> a symptom of *readings living in prose*, not of two files existing. But which document
> owns a KPI is a governance call, not a drafting one.
>
> One thing to note whichever way it goes: **`docs/kpis.md` at the repository root does not
> exist.** The live path is `packages/confit/docs/kpis.md`; a spec, ticket or memory citing
> the bare root path is a stale path, exactly as the oracle spec's "Doc homes" note records
> for `docs/known-limitations.md`.
>
> *Binds:* `packages/confit/docs/kpis.md`, `packages/confit/docs/properties.md`, section 5
> in full, and the oracle spec's claim: fit-serving-oracle.

---

## ASK index

| ask | question | binds |
|---|---|---|
| ask: acceptance-target | is there an acceptance-rate target, and does the ladder ratchet? | goal: growing-accepted-surface, kpi: acceptance-rate, kpi: ladder-ratchet |
| ask: next-query-classes | which query classes are next, in what order? | goal: growing-accepted-surface, four `exclusion:` rows |
| ask: exclusion-ratification | does the ten-row ledger bind, silent triggers and all? | every `exclusion:` slug |
| ask: kpi-set-change | adopt any of the six proposed KPIs, and as what kind? | all six `kpi:` slugs |
| ask: bench-baseline-flip | the serving-latency sign flip: regression, or a changed baseline? | goal: request-latency-budget, kpi: bench-refresh-cadence |
| ask: kpis-absorb-or-defer | does this document absorb kpis.md or defer to it? | kpis.md, section 5 |

Two questions this document deliberately does **not** ask, because they are already open in
the oracle spec and forking them would split the answer: the corpus match count's ratchet
(ask: match-count-ratchet — claim: corpus-match-today is fresh evidence for it) and the
undocumented refusal prefixes (ticket: clean-prefix-reconcile — claim: refusal-prefix-share
adds two families to it).

One finding recorded and **not** acted on here, because the fence for this task is one
document: the seed-1804 `DIVERGE_VALUE` in claim: campaign-verdicts-today is a live parity
defect and wants an xfail-strict pin plus a ticket.
