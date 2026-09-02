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
*2026-09-02* was produced there. **The build is a precondition, not a detail:** a fresh
checkout ships no compiled extension, so every measured number in sections 2 and 5 is
unreproducible until that build completes — and `benchmarks/bench_serving.py:31-35` records
that the *wrong* build silently corrupts the engine rows rather than failing. One number
here could **not** be produced in this environment and says so where it appears: the L3
Spark floor needs `pyspark`, which is not installed (claim: dialect-floors-today).

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
a `ValueError` naming the construct. Nothing is approximated, silently dropped, or widened
at inference time. This is the load-bearing goal: everything in section 2 follows from it.

**The third mode exists and is enumerated — that is the whole difference.** The absolute
above is "no *unenumerated* third mode", not "no third mode", and writing it the strong
way would make this document false on its own evidence. The enumeration is the oracle
spec's divergence ledger
(`packages/confit/docs/oracle/07-the-divergence-ledger.md`), whose rows *serve* with a
consciously different surface. Four bear on this goal today:

- divergence: decimal-literal-typing — a bare DECIMAL literal serves `double` where the
  oracle emits `decimal128(p,s)`; measured 2026-09-02, `SELECT 1.5 AS o0 FROM __THIS__`
  builds and returns `double`. Severity 2, *unruled*.
- divergence: narrow-lane-overflow — an overflowing narrow lane serves the i64 value on
  the row path until the trap phase lands. Severity 3, `PINNED` until-fixed.
- divergence: schema-qualifiers — `s1.t1` on a non-existent schema serves here and raises
  `schema "x" does not exist` on DuckDB. Severity 3, `PINNED`.
- divergence: dedup-on-both-sides — duplicate output names carry the client-contract
  rename. Contract choice, `PINNED`.

Which rows are in force, and at what severity, is the ledger's status column to say, never
this document's (section 6).
*Enforced-by:* build-time refusal at `DuckDBInferFn(...)` construction throughout
`packages/confit/src/specializer/`; the ledger for the enumerated exceptions.
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
*Enforced-by:* **nothing — `Unverified`.** No gate, floor or pin bounds serving latency;
`benchmarks/` prints a table and asserts a three-way *parity* gate, not a time. Read the
front matter's rule against goals-defined-by-current-measurement as binding here: the
number below is evidence for the regime, and a regression that moved it would not fail
anything.
*Verified-by:* measured 2026-09-02 (claim: bench-is-stale): `spec` p50 at n=1 is
2,300-8,800 ns per call across the five `benchmarks/` scenarios, against 5.3-13.1 ms for
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
— parity outside the enumerated divergence ledger is fixed at 100% by
goal: two-outcome-contract — it is **acceptance**: how much real SQL builds instead of
refusing. Progress is queries moving from REFUSED to served,
never into a wrong answer. Section 2 is this goal restated with numbers.
*Enforced-by:* **two of the four numbers in section 2, and only two.** The dialect floors
gate (`SUPPORTED_FLOOR = 288` at `test_dialect_corpus_gate.py:37`, `SPARK_MATCH_FLOOR =
260` at `test_dialect_cross_engine_gate.py:50`). The corpus match count is *printed* and
never asserted — `test_corpus_replay.py:180-184` prints it, `:186` asserts only `not
fails` — and the campaign acceptance rate has no gate at all. A count records a goal; it
does not make one hold. That gap is what ask: acceptance-target and kpi: ladder-ratchet
are about.
*Verified-by:* `packages/confit/docs/kpis.md:100-116` (D1, "Progress = moving queries from
REFUSED to MARGINALIZED (never to FAILED)"); the four line reads above, 2026-09-02.

---

## 2. Parity is a control; acceptance is the goal

The reframe this document exists to make explicit. Because the contract is
bit-exact-or-refuse-by-name (goal: two-outcome-contract), **parity on the accepted surface
outside the divergence ledger is a control fixed at 100%**: a parity violation there is a
bug to fix, never a dial to trade. Asking "how much oracle parity do we want" has exactly
one answer, and it is not interesting. The goal-shaped question is **which queries are
accepted at all**, and how fast that set grows.

Three verdicts carry the distinction, and the oracle spec defines them
(`packages/confit/docs/oracle/04-verdicts-agreement-abstention-refusal.md`):

| verdict | means | counts as |
|---|---|---|
| `AGREE` | ours == optimizer-off DuckDB == optimizer-on DuckDB | the only thing counted as coverage (claim: coverage-accounting) |
| `REFUSED` | confit refused at build; the case never entered the comparison | **not** a finding and **not** coverage — absorbed into a histogram (claim: refusal-absorb). This is the number this section is about |
| `UNSHIPPED` | the oracle's answer has a width we have not shipped, so *nothing was compared* | neither agreement nor finding; reported in its own section (claim: unshipped-verdict) |

Three of eleven, and the three are not a partition. The full verdict space is
`fuzz.oracle.KINDS` (eleven), plus two the *runner* produces rather than the oracle —
`TIMEOUT` and `PANIC`, both in `fuzz.runner.INTERESTING`, both reaching `findings.jsonl`
by the same path as every other verdict. The oracle spec is explicit that any statement
about what a campaign reports has to include them
(`04-verdicts-agreement-abstention-refusal.md:27-29`), so the tables below say when they
were zero rather than leaving them out.

`REFUSED` is the shape of "not accepted". `UNSHIPPED` is **not**: an unshipped case
*builds and serves*, and only the comparison is withheld — which is why it sits inside the
acceptance numerator below and is the divergence ledger's business, not the refusal
histogram's. The two retire differently all the same: a refusal retires by a decision to
serve the construct, an unshipped width by the width landing.

### 2.1 The accepted surface, measured

**claim: corpus-match-today.** **[FACT]** Of the 678 statements mined from DuckDB's own
test suite, **547 replay bit-exact, 131 refuse cleanly, 0 FAIL** — measured 2026-09-02 by
`uv run pytest packages/confit/tests/test_corpus_replay.py -s`. **550** is quoted at six
unhedged sites across *five* documents (not six documents), and the oracle spec's ladder
records 550 as a genuine earlier reading (`53 -> 395 -> 505 -> 511 -> 529 -> 546 -> 550`),
so the ungated match count has moved **down by three** since it was last written down and
nothing noticed. Zero FAILs is the gate and it holds; the match count is deliberately
ungated. One note on the cleanup's size, because ask: match-count-ratchet turns partly on
it: a bare `grep -rn 550 --include=*.md` on 2026-09-02 also finds the number in
`oracle/09-version-bumps-and-mutability.md:56`, a second occurrence in
`reports/confit-architecture.md:30`, and `docs/specs/2026-08-13-dialect-logical-plan-design.md:265`
and `:298` — so the six *unhedged current-state* sites the correction enumerates are a
subset of the occurrences a remediation would have to walk.
*Verified-by:* the run above; `packages/confit/tests/test_corpus_replay.py:171-190` (the
zero-FAIL gate); `packages/confit/docs/oracle/` claim: zero-fails-gate and its correction,
which names those six sites. This measurement is fresh evidence for the oracle spec's
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
measured 2026-09-02 by `python -m fuzz.runner --seed 0 --n 2000 --workers 8 --timeout 20`
(the flags matter and are not the defaults — `runner.py:213-214` defaults to `--workers 4
--timeout 30`), and re-run twice with identical counts. **Corroboration, not
independence:** the second route was `fuzz.oracle.run_case_json`, which *is* what the
runner's workers call (`fuzz/worker.py` is nine lines around it), so the two runs share
every line except the subprocess and timeout wrapper. That wrapper is exactly the layer
that can turn a verdict into `TIMEOUT` under load, so it is the layer a second run most
needs to exercise, and this pair does not.

| verdict | count | share |
|---|---|---|
| `AGREE` | 1013 | 50.7% |
| `REFUSED` | 944 | 47.2% |
| `AGREE_TRAP` | 21 | 1.1% |
| `UNSHIPPED` | 14 | 0.7% |
| `DIVERGE_OPT` | 7 | 0.35% |
| `DIVERGE_VALUE` | 1 | 0.05% |

`TIMEOUT` and `PANIC` were **0** on this run, and that is a reading rather than a
property: a machine under load moves a case from its true verdict into `TIMEOUT`, which
moves acceptance by one. Any re-run that reports them non-zero has not found a defect, it
has found load — the fix is to re-run the named seed alone, not to re-record the table.

**Acceptance** — cases that **built rather than refused**, which is the only definition the
arithmetic supports — is **1056/2000 = 52.8%**. That numerator deliberately contains the 14
`UNSHIPPED` (they build and serve, only the comparison is withheld), the 21 `AGREE_TRAP`,
and the one live `DIVERGE_VALUE`: on this definition the seed-1804 parity defect below
counts as accepted, which is correct for a *scope* metric and is the reason acceptance can
never stand in for parity. All 14 `UNSHIPPED` are class `decimals`
(exclusion: unshipped-decimal-arithmetic). All 7 `DIVERGE_OPT` are
exclusion: optimizer-on-answers' standing cost — **reported findings, not an accepted
class**, which is the oracle spec's claim: contract-surface-gap and binds here. The one
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
schema-mode** curated. Verified fresh 2026-09-02, and it takes **two** tests, not one:
`_corpus_test.py::test_progression_totals` (`:232-239`) pins the totals — `len(MINED) ==
22`, the two mined buckets summing to 22, and the three curated lengths — while the
**11/11 split itself** is pinned only by `::test_mined_corpus_scoreboard` (`:197-207`,
`counts == MINED_SCOREBOARD`). A drift to 12/10 passes the first and fails the second, so
citing the totals test alone for the split would name a gate that cannot catch it. That
ladder is `sql_transform`'s, not confit's (goal: engine-half-only);
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
> The campaign says which of these the *generator* reaches, not which your users need. The
> measured top of the refusal histogram, seeds 0-1999, 2026-09-02: `WITH` 104 of 944,
> `comparison on BOOLEAN` 86, `bind error: bad integer literal` 43, `shape=map` WHERE-drop
> 42, `udf udf0: a width-1 list return` 42, `modifier on scalar call abs` 39, `QUALIFY` 36,
> `DISTINCT` 30, `ORDER BY` 23. Only the first is a whole-relation shape near the top;
> `QUALIFY` ranks **seventh**, not third. `comparison on BOOLEAN` — the second-largest
> class — belongs to no exclusion row at all; see ask: exclusion-ratification (2). That is
> a fact about the grammar, not a demand signal.
>
> *Binds:* goal: growing-accepted-surface, exclusion: unshipped-decimal-arithmetic,
> exclusion: wide-integer-lanes, exclusion: non-scalar-values,
> exclusion: multiplicity-by-default.

---

## 3. What we exclude for now

**Every row in this section is `[PROPOSED]`.** The marker is stated once here rather than
ten times below, and it is not decoration: the *behaviour* each row describes is in force
and measured, but the **ground assigned**, the **retirement trigger named** and the
**ledger's completeness** are exactly what ask: exclusion-ratification puts to the owner.
Read the declarative voice below as "this is what the evidence says the rule is", never as
"this is ruled". A reader grepping for `[PROPOSED]` should find this section.

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
multiple statements, table functions in `FROM`, `rowid`, `FULL OUTER JOIN`, **`QUALIFY`**
(`frontend.rs:284`), **named window definitions** (`WINDOW`, `:314`) and **window/aggregate
modifiers on a scalar call** — `OVER`, `FILTER`, `IGNORE NULLS`, `WITHIN GROUP`
(`:6289-6299`). The last three were absent from this enumeration and are not small:
`QUALIFY` is 36 of 944 campaign refusals and the modifier class 39, and `abs(k) OVER ()`
was a silently-dropped modifier on master until it was made to refuse — measured
2026-09-02, it now raises `unsupported: modifier on scalar call abs`.
*Why:* scope-by-product-decision. Their output shape is not one-row-in / 0..N-out, so they
are not row-at-a-time feature transforms. Measured share: `WITH` alone is 104 of 944
campaign refusals, `QUALIFY` 36, `DISTINCT` 30, `ORDER BY` 23.
*Retirement trigger:* none intended. The carve-out already exists and is not a retirement:
a **static-tables-only** query is evaluated once by DuckDB at build and frozen, so
aggregation and `ORDER BY` serve there — with row limits refused, because which rows
survive a limit is not a function of the query (measured: four different answers across
twelve connections). **The carve-out has a hole its source names and this row used to
drop:** only a *row limit* refuses, and `ORDER BY` does not fix ties (a tie fed from a
`GROUP BY` flipped in 20 runs, `known-limitations.md:116-117`). So a static-only
tie-producing `ORDER BY` builds and freezes whichever order that build's DuckDB run
happened to get — two builds of the same function can disagree with each other, which is
goal: serving-without-skew's failure in its build-to-build face rather than its
train-to-serve one. Nothing refuses it today.
*Rings:* nothing rings — there is no trigger. Lifting one breaks
`packages/confit/tests/test_known_limitations.py` — which is the executable twin of
`known-limitations.md`, **not of this row**. Its docstring says so (`:1-7`, "Section
numbers mirror the document"), it holds 14 test functions, and its whole-relation
parameterization (`:98-117`) covers nine constructs: aggregates, `GROUP BY`, `ORDER BY`,
`LIMIT`, `DISTINCT`, `WITH`, `UNION`, `rowid`, `FULL OUTER JOIN`. `HAVING`,
`OFFSET`/`FETCH`/`TOP`, `INTERSECT`/`EXCEPT`, subqueries, multiple statements, table
functions and `QUALIFY` — all enumerated above — have no twin there. The oracle spec's
claim: doc-twin-totality measured this across five other sites and ask: doc-twin-overstatement
is open on it; this document does not become a sixth site.
*Verified-by:* `packages/confit/docs/known-limitations.md:95-120`;
`packages/confit/tests/test_known_limitations.py:1-7, :98-117` (counted 2026-09-02).

**exclusion: per-row-general-work.** Non-constant regex patterns, replacement strings,
regex options and extract-group indexes; anything that would compile or bind per row.
*Why:* specialization-inherent, and the direct negation of goal: pack-time-only-work.
DuckDB compiles regexes per row; we compile at prepare.
*Retirement trigger:* a decision to abandon compile-once for this construct. Effectively
permanent — retiring it contradicts a goal, so it would be a goal change first.
*Rings:* nothing rings.
*Verified-by:* `packages/confit/docs/known-limitations.md:74-75`.

**exclusion: unshipped-decimal-arithmetic.** What is **excluded** is narrower than the
heading suggests, so it is worth stating what still serves: comparisons, joins, `CAST(d AS
DOUBLE)` and `SELECT *` over decimals all serve (`known-limitations.md:148`), and decimal
*statics* serve exactly in an i128 lane. The exclusion is **expressions** over DECIMAL —
arithmetic, `CAST` to anything but `DOUBLE`, and `COALESCE`/`CASE`/`greatest` unifying a
decimal with a non-identical type.
**A DECIMAL literal is not excluded — it *serves*, at a different width.** Measured
2026-09-02, `SELECT 1.5 AS o0 FROM __THIS__` builds and returns `double` where the oracle
returns `decimal128(2,1)`; the campaign's `UNSHIPPED` verdict means built-and-not-compared,
never refused. The visible consequence is that `CAST(-2.5 AS BIGINT)` on a bare literal
rounds half-to-even here and half-away-from-zero there. And the real split is not
literal-vs-static-column but **row-path vs static-only path**: the same literal reached
through a static-tables-only query never prepares, so DuckDB evaluates it and it comes back
`decimal128(2,1)` and `AGREE`
(`test_fuzz_smoke.py::test_the_static_only_leg_has_no_unshipped_width_to_classify`).
**This row overlaps the divergence ledger, and here is the pointer this document promised
for each such row:** the served-width fact is divergence: decimal-literal-typing and the
rounding fact is divergence: decimal-cast-rounding, both `unruled` there pending
ask: float-tolerance-list. The *scope* decision is this row's; the *symptom's* status is
theirs, and this row does not settle it.
*Why:* scope-by-product-decision, pending the lattice phase. Each of the excluded
expressions used to serve a silently wrong double, which is why they refuse now rather than
approximate.
*Retirement trigger:* **decimal arithmetic ships** (m-8 lattice phase 5).
*Rings:* **yes, loudly, and this is the one exclusion with a real bell.** The campaign
classifies these cases `UNSHIPPED` instead of comparing them, `fuzz.oracle._type_delta`
carries exactly one arm (decimal-against-float64) which is deleted when the feature lands,
and the runner prints the bucket in a section of its own — an empty section means either
the feature shipped or the grammar stopped reaching it, and both are worth seeing.
Measured 2026-09-02: 14 of 2000 seeds, all class `decimals`.
*Verified-by:* `packages/confit/docs/known-limitations.md:138-174`;
`packages/confit/fuzz/oracle.py:124-145` (the note), `:498-505` (the arm),
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
*Rings:* partially, and **more loudly for the unsigned family than an earlier draft of this
row claimed**. `test_integer_widths.py::test_unserved_static_type_refuses_by_name`
(`:856-871`) parameterizes `uint8`/`uint16`/`uint32`/`uint64` (and `float32`) and asserts
the refusal names the column and the type — so shipping any unsigned *static* width turns
that test red by name, on the day it ships. `test_known_limitations.py:180-192` pins the
uint64 static the same way. What still rings nothing: **HUGEINT**, unsigned *row* columns,
and the narrow-lane trap phase.
*Verified-by:* `packages/confit/docs/known-limitations.md:122-136`, `:175-187`;
`packages/confit/tests/test_integer_widths.py:856-871` (read 2026-09-02).

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
*Verified-by:* `packages/confit/docs/known-limitations.md:149-165`, `:207` — note the
`decimal256` and decimal-row-column facts this row names are at `:149-150`, and `:166`
opens the DECIMAL-literals bullet, which belongs to
exclusion: unshipped-decimal-arithmetic and not here.

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
*Rings:* **for the string budget only.**
`known_divergences/test_string_budget.py::test_a_budget_breaking_literal_count_refuses`
(`:145-146`) asserts the refusal with `pytest.raises(ValueError, match="builder|GiB")`, so
that ceiling rings on a *change*. The **Arrow batch ceiling rings nothing**: measured
2026-09-02, `test_arrow_boundary.py` is 120 lines holding four tests
(`:38`, `:63`, `:78`, `:100`), none of which touches the ceiling, and its only mention of
it is a **prose comment** at `:33-35`. A `Verified-by` pointing at a comment names nothing
that would fail, which is this document's own definition of not-verified — so:
**`Unverified` for the Arrow half.** This row is a live instance of
ask: exclusion-ratification (1), inherited from the same pointer in
`oracle/07-the-divergence-ledger.md:147`.
*Verified-by:* `packages/confit/docs/known-limitations.md:205`;
`packages/confit/tests/known_divergences/test_string_budget.py:145-146`;
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
> Ten rows above, all `[PROPOSED]`, assembled from `known-limitations.md`, the engine's
> refusal sites and the campaign's refusal histogram. Ratifying them means three things
> bind: the **ground** assigned to each (specialization-inherent vs
> scope-by-product-decision vs resource, which is what makes "we refuse" auditable), the
> **retirement trigger** named, and how many rings nothing.
>
> Three specific pieces I would like ruled on rather than assumed:
>
> **(1) Are the silent rows acceptable? Counted honestly, it is seven of ten, not six.**
> Reading every `Rings:` line: exclusion: whole-relation-shapes, per-row-general-work,
> non-scalar-values and statistics-dependent-kernels ring nothing at all;
> exclusion: multiplicity-by-default rings nothing *for its retirement*;
> exclusion: resource-ceilings rings on a change and not on a retirement, and its **Arrow
> half rings nothing at all** — a `Verified-by` that pointed at a prose comment; and
> exclusion: wide-integer-lanes rings partially, but **more than an earlier draft said**:
> its unsigned-static half is pinned by name in `test_integer_widths.py:856-871`. So the
> count moved in both directions on re-measurement, which is itself the argument for
> ratifying an inventory rather than inheriting one. Silence is fine where the trigger is
> "never"; it is a live gap for exclusion: non-scalar-values and
> exclusion: multiplicity-by-default, whose triggers are real scheduled-ish work. The
> pattern that *does* work is exclusion: unshipped-decimal-arithmetic's: a classified
> verdict, a single code arm deleted on landing, and a report section that empties.
>
> **(2) Is the enumeration complete? Measurably not, and here is the largest hole.**
> `unsupported: comparison on BOOLEAN` (`specializer/frontend.rs:5322`) is the
> **second-largest refusal class in the campaign** — 86 of 944, 9.1%, behind only `WITH` —
> and it appears in **no** exclusion row and **nowhere** in `known-limitations.md`
> (`grep -c BOOLEAN` there returns 0, measured 2026-09-02). `unsupported: modifier on
> scalar call abs` (`:6296`, 39 refusals) reached the ledger only by being added to
> exclusion: whole-relation-shapes in this pass. By `known-limitations.md:284`'s own rule a
> refusal in code and in no document is a bookkeeping bug to file — these are the biggest
> instances this document's own measurement surfaced.
>
> The census behind the gap, and it needs splitting to be honest: 281 `unsup(...)` call
> sites across nine source files (measured 2026-09-02), but **only ~166 of them are this
> ledger's business**. `specializer/frontend.rs` 144 and `specializer/retrans.rs` 22 raise
> `PrepareError` — what `DuckDBInferFn` refuses, which is what these rows are about (two of
> the 166 are the `fn unsup` definitions themselves, `frontend.rs:64` and `retrans.rs:11`).
> The other 115 live under `src/dialect/` (`duckdb.rs` 84, `plan.rs` 13, `bigquery.rs` 7,
> `spark.rs` 6, `printer.rs` 2, `ty.rs` 2, `mod.rs` 1) and raise `DialectError::Unsupported`
> — a different error type on the SQL-to-plan-to-SQL translation surface feeding the L2/L3
> cross-engine gates (`src/dialect/mod.rs:19-24`), which claim: dialect-gate-oracle scopes
> out. So the gap the owner is being asked to rule on is ~164 call sites against 10 rows,
> not 281 against 10. No mechanism maps either set to a ledger row.
>
> **(3) Do the three grounds cover what actually refuses?** They classify *scope*
> decisions, and the histogram holds a class that is not one: `udf '<name>': a width-1 list
> return is a scalar` (`src/duckdb/mod.rs:606-609`) is **63 of 944 refusals** and is neither
> specialization-inherent, nor scope-by-product-decision, nor resource — the caller declared
> their UDF wrong. Adding a message prefix would not classify it, because the taxonomy has
> no slot for **caller-declaration errors**. They nevertheless sit in the same `REFUSED`
> bucket that feeds the 52.8% in claim: campaign-verdicts-today. Either a fourth ground, or
> a rule that such refusals are excluded from the acceptance denominator — both are your
> call, and the second is the one kpi: acceptance-rate depends on.
>
> *Binds:* every `exclusion:` slug in this section, kpi: acceptance-rate, and
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
*Verified-by:* `packages/confit/docs/kpis.md:58-61` (C2's Enforced bullet, which names
`test_udfs.py` (`udf_check`) among its enforcing suites — `:44-56` is C2's "Which DuckDB"
prose and names no suite); `test_udfs.py::test_a_udf_may_not_take_a_builtin_name`.

**claim: sklearn-is-the-reference.** On the surface DuckDB cannot run, an independent
sklearn reference plays the role optimizer-off DuckDB plays for SQL — and it is a
**reference**, not the oracle, which is why it comes with a named bound instead of bit
equality everywhere. The bound differs by family, and **there are three bounds in force,
not one** — naming only the loosest would be the quiet loosening kpis.md forbids:

| surface | bound | where |
|---|---|---|
| tree scoring | **bit-exact** against `sklearn.predict` | `_trees_test.py::test_matches_sklearn_bit_exactly` |
| bare `StandardScaler` transformer columns | **`rtol=1e-12`** | `_transformers_test.py:67, :81, :204, :271` |
| `Pipeline([StandardScaler, PCA])` columns | **`rtol=1e-9`** | `_transformers_test.py:109, :127` |
| the campaign's sklearn metamorphic leg | **absolute `1e-9`**, not a relative tolerance | `fuzz/oracle.py:845` — `abs(o - p) > 1e-9` |

Four of the six `assert_allclose` sites in the transformer file are the tighter `1e-12`;
the campaign's `1e-9` shares a numeral with the transformer bound and not a meaning.
*Enforced-by:* the four rows above, read 2026-09-02;
`packages/sql-transform/sql_transform/_transformers_test.py::_reference` (`:34`) is the
clone-per-group reference all the transformer rows compare against.
*Verified-by:* `packages/confit/docs/kpis.md:76-87` (C4, the control this is — note C4's
own text names **no** tolerance, so the bounds above are read from the tests, not from the
KPI); `packages/confit/docs/oracle/` claim: metamorphic-self-legs (the `1e-9` leg, itself
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

**Three of the six candidates in 5.3 are bars, not drives, and one of them lands on C5.**
Saying otherwise here would be an unforced error in the one paragraph whose job is to show
the standing law is respected, so it is said plainly instead:
kpi: named-refusal-share is proposed as a **control at 100%** over refusal message quality,
and C5's own text is "refuses at construction with an error naming the construct"
(`kpis.md:89-94`) — so it is a second bar over C5's clause, proposed at **88.7%**, which
`kpis.md:12` calls "not a control, it is an unacknowledged trade-off". kpi: ladder-ratchet
is a never-decreases rule and kpi: bench-refresh-cadence a staleness bound; neither is a
drive. **None of the six may be adopted in a form that weakens C1-C5**, and the sequence
that respects the law for the one that touches C5 is spelled out in its own entry: drive
first, control only after the gap is closed. Nothing here is adopted; ask: kpi-set-change
is the door.

### 5.2 What is measured today

**claim: kpi-pointers-resolve.** **[FACT]** Every enforcement pointer under the five
control KPIs in `packages/confit/docs/kpis.md` resolves on master `2ba96e5`, checked
2026-09-02: C1 `_projection_test.py::gate` (`:34`) and its `MARGINALIZE_FUZZ_N` seeded
differential (`:374-376`, seed 20260729); C2 the `test_duckdb_*.py` wave suites,
`test_params_joins.py`, `test_udfs.py::udf_check` (`:240`), `known-limitations.md`, and
`src/specializer/exec/tests.rs`; C3 `_serving_test.py::serve_gate` (`:29`); C4
`_transformers_test.py::_reference` (`:34`); C5 `_corpus_test.py` and its three-outcome
FAILED-must-be-empty rule. Two honest notes:

**C1 is enforced two orders of magnitude shallower than its text reads.** C1 says
"1,500-2,000-case runs at each widening loop" (`kpis.md:38-39`), but
`_projection_test.py:375` is `n = int(os.environ.get("MARGINALIZE_FUZZ_N", "25"))` — the
deep run is opt-in and the standing gate is 25. Under `kpis.md:12` a control that only
holds at its written depth when someone sets an environment variable is "an unacknowledged
trade-off", and the remedy is a decision, not an edit: **either correct C1's text to 25, or
raise the default and pay the runtime**. That is the owner's, and it is carried to the
close of this document rather than dropped here.

**D1's pins are fresh, and it takes two tests to say so** — `test_progression_totals`
(`:232-239`) pins the totals, `test_mined_corpus_scoreboard` (`:197-207`) pins the 11/11
split; see §2.2.
*Verified-by:* the file and line reads above, plus a passing run of the
`_corpus_test.py` suite on 2026-09-02.

**claim: bench-is-stale.** **[FACT]** D2's serving-latency table is dated **2026-08-04**
(`kpis.md:118`) and is **389 commits** behind `2ba96e5`. It carries no commit of its own —
`a6fa318` appears in kpis.md only at `:143`, introducing the *transformer*-path tables, and
attaching it to the pure-SQL table would be a misreading. Worse than age: D2's headline
claim — "~1.5-1.7x faster than a handwritten Python microservice twin" — compares against a
harness row that **no longer exists**. The record of that retirement is in the harness, not
in kpis.md (which has zero occurrences of `python_dict`, `spec_dict` or `pydantic`):
`benchmarks/bench_serving.py:22-25`, "the old typed-model `python` and `spec_dict` rows
retired with the pydantic surface — dict IS the output now". Re-measured 2026-09-02 (`uv
run python -m benchmarks.bench_serving`, parity gate green on all five scenarios), the
comparable pair is `spec` against the surviving `python_dict` row, p50 ns at n=1:

| scenario | spec | python_dict | ratio | duckdb per call | 2026-08-04 baseline? |
|---|---|---|---|---|---|
| titanic | 3,400 | 2,200 | 1.55x slower | 6,979,500 | yes |
| house_prices | 5,100 | 4,400 | 1.16x slower | 11,260,900 | yes |
| fraud_txn | 5,400 | 4,800 | 1.13x slower | 13,106,300 | yes |
| store_sales | 8,800 | 4,300 | 2.05x slower | 12,262,900 | yes |
| feature_bundle | 2,300 | 1,400 | 1.64x slower | 5,333,400 | **no** |

Two corrections to how this was first written down, both mattering to the conclusion.
**(i)** An earlier reading recorded `feature_bundle`'s `python_dict` cell as **2,300** and
drew "level on the fifth" from it. It reads **1,400** here and on every repeat run of
2026-09-02, and 2,300 is the *titanic* `python_dict` value one row up — a transcription
slip, most likely. The correct statement is **slower on all five**, and the row that looked
like a tie carries the **second-worst** ratio in the table. **(ii)** D2's pure-SQL table
(`kpis.md:126-131`) has **four** rows; `feature_bundle` has no 2026-08-04 baseline at all.
The comparison is four-against-four plus one new scenario, so "uniform across five" borrows
strength from a row with nothing to be uniform with.

The three-orders-of-magnitude gap to DuckDB-per-call (goal: request-latency-budget) is
intact. The sign of the Python comparison is not: the recorded claim says faster, today's
run says slower everywhere. **This document does not call that a regression** — the
baseline row changed identity (plain dicts out, where the old row returned pydantic
models), the machine is not the 2026-08-04 machine, and kpis.md warns absolute numbers
drift with load. That warning is not theoretical here either: `store_sales`' `spec` cell
read 5,600 and 8,800 ns in two runs on the same machine on the same day, a 57% spread,
which is exactly why kpi: bench-refresh-cadence proposes recording a *ratio*. It is a sign
flip in a claim nobody re-ran for a month. See ask: bench-baseline-flip.

**claim: refusal-prefix-share.** **[FACT]** Refusal *quality* is measurable today and has
never been measured. Over seeds 0-1999, 2026-09-02: of 944 refusals, **837 (88.7%)** carry
one of the three documented prefixes (`unsupported:` / `parse error:` / `bind error:`);
**107 (11.3%)** carry none. **688 (72.9%)** fall in the corpus gate's narrower `_CLEAN`
set, which excludes `bind error:` entirely — and admits two families that are not prefixes
at all: `_CLEAN` is `("unsupported:", "parse error:", "duplicate map key", "NULL in value
column")` (`test_corpus_replay.py:36`), whose last two are *substrings* and appear in
neither `known-limitations.md:279-282`'s three-prefix list nor anywhere else in the docs.
The corpus gate's own definition of clean is already wider than the documentation.

**What 88.7% measures, and what it does not.** It measures *prefix presence*, not that the
message names a construct — those are different properties, and the weaker one is the one
with a number. A correlated subquery is the demonstration: it refuses, correctly, but
through the frontend's catch-all `other => Err(unsup(format!("expression: {other}")))`
(`frontend.rs:3551`), which echoes the AST text rather than naming the construct, and there
is no subquery-specific refusal site to find. It counts inside the 837 and inside the 688
all the same. kpi: named-refusal-share below is defined on the *stronger* property, so
88.7% is a floor for it, not a reading of it.

The undocumented 107 are three families, and
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
editing the generator — and that is not a hypothetical, it is already true of the number
printed. `fuzz/gen.py:1060` reads `w = rng.choice([1, 2, 3])  # width-1 list must REFUSE`:
the generator deliberately emits UDF return shapes it knows are rejected, and those are
**63 of 944 refusals**, roughly **3.2 percentage points** of the 47.2% refusal rate.
Changing that one `rng.choice` moves acceptance by about three points with no engine change
at all. Adopting it as a *drive* without claim: coverage-denominator's triple-axis work
invites optimizing the wrong thing; adopting it as a *reported statistic* costs nothing.
Either way ask: exclusion-ratification (3) has to settle first whether a caller-declaration
error belongs in the denominator.

**kpi: findings-per-campaign.** **[PROPOSED]** An explicit **zero-control**: findings per
N seeds, counted **beside** `DIVERGE_OPT` rather than with it. The wording matters: the
oracle spec's claim: contract-surface-gap is explicit that `DIVERGE_OPT` "stays a reported
finding ... rather than an accepted class", so this KPI may hold it out of the *bar* — a
zero-control over a class we knowingly tolerate would be red on day one — but never out of
the *report*. Today: **1 per 2000** on the bar (the seed-1804 NaN-sign case) beside **7**
`DIVERGE_OPT`, and `TIMEOUT`/`PANIC` at 0, which a control also has to name or a loaded
machine reads as a finding.
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

**kpi: named-refusal-share.** **[PROPOSED]** A **control** at 100% — and it is the one
candidate that is a *bar*, over C5's own clause; 5.1 says so rather than pretending
otherwise. The property: every refusal carries a documented, actionable prefix **and names
the construct**. Today's **88.7%** (claim: refusal-prefix-share) measures only the prefix
half, so it is a **floor** on the real number, not the real number: the catch-all
`expression: {other}` site passes the prefix check while naming nothing.
*Measurement that backs it:* the measurement in 5.2, which is a 40-line script over the
campaign's own verdicts — for the prefix half. The construct-naming half has **no
measurement today** and would need one before the bar could be set.
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
*Two scope notes before this one is adopted, because both change what it costs.* The
sql-transform ladder is **not confit's** (goal: engine-half-only, and §2.2 says so four
lines after citing it) — a confit document proposing a gate over the other package's
admission ladder is annexation unless the owner rules the KPI cross-package. And the L3
Spark floor of **260 cannot be read on the reference environment**: `pyspark` is not
installed here and the fixture fails loudly rather than skipping (claim: dialect-floors-today),
so ratcheting it means naming the environment that checks it.
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
> **1.13-2.05x slower** than the twin row that still exists (claim: bench-is-stale), on all
> five scenarios — though only **four** of them have a 2026-08-04 baseline to have flipped
> from. Four readings, and the cheapest one to rule out is the one an earlier draft of this
> ASK left out:
>
> **(a) Baseline changed identity.** The old `python` row returned pydantic models; the
> surviving `python_dict` row returns plain dicts, which is cheaper. The magnitudes fit:
> titanic's Python twin read 6,000 ns in D2 (`kpis.md:128`) and 2,200 ns today. If that
> accounts for the whole delta, D2's prose is simply obsolete and the fix is an edit.
>
> **(b) Real regression.** 389 commits landed since the measurement, including the Arrow
> schema API, the lane/slot seams and join-key work. Ruling this out costs one bisect
> against `a6fa318` on the same machine.
>
> **(c) Machine noise.** Other agents active. Not as cheap to dismiss as it looks:
> `store_sales`' `spec` cell read 5,600 and 8,800 ns in two runs the same day.
>
> **(d) A stale or debug wheel — and the harness warns about exactly this symptom, in
> capitals.** `benchmarks/bench_serving.py:31-35`: "rebuild the wheel first (`uv run
> --reinstall-package confit python -c pass`) — a stale wheel inflates ONLY the engine rows
> and once produced a phantom 7x regression (caught by bisection, 2026-07-26)". Engine rows
> uniformly slower with the Python row unchanged **is** the stale-wheel signature, it has
> happened in this repo before, and it is the cheapest of the four to rule out. The
> measurement environment above names `uv run maturin develop --release`, which is not the
> harness's own recommended command; ruling (d) out costs one re-run after
> `--reinstall-package`.
>
> I did not bisect — the fence for this task is one document. What I can say is that (c)
> alone is unlikely to produce a uniform sign flip, that (d) is untested and cheapest, and
> that the answer changes what goal: request-latency-budget is worth as a claim.
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
oracle spec wins on *correctness* questions and this document wins on *scope* questions.
Concretely, and it is tested by three places above: section 1 **names** four divergence
rows as the enumerated exception to the two-outcome contract and takes their status from
the ledger's own column; exclusion: unshipped-decimal-arithmetic **names** the two rows it
overlaps and leaves them `unruled` where the ledger leaves them; and
claim: campaign-verdicts-today defers to claim: contract-surface-gap on whether
`DIVERGE_OPT` is an accepted class (it is not). Citing a row is deference; restating its
verdict would be re-litigation, and section 2 does not redefine a verdict either.

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
> merged PRs — absorbing it invalidates line-anchored citations across the tree. A second
> cost this option looks cheaper than it is without: **C1, C3, C4, C5 and D1 are all
> enforced in `packages/sql-transform`** (`_projection_test.py`, `_serving_test.py`,
> `_transformers_test.py`, `_corpus_test.py`) — only C2 is confit's. Absorbing would import
> the other package's KPI half wholesale into a document whose §1 says it is only about
> confit's half (goal: engine-half-only).
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
| ask: exclusion-ratification | does the ten-row ledger bind — silent triggers, the classes it misses, and the ground it has no slot for? | every `exclusion:` slug, kpi: acceptance-rate |
| ask: kpi-set-change | adopt any of the six proposed KPIs, and as what kind? | all six `kpi:` slugs |
| ask: bench-baseline-flip | the serving-latency sign flip: regression, or a changed baseline? | goal: request-latency-budget, kpi: bench-refresh-cadence |
| ask: kpis-absorb-or-defer | does this document absorb kpis.md or defer to it? | kpis.md, section 5 |

Three questions this document deliberately does **not** ask, because they are already open
in the oracle spec and forking them would split the answer: the corpus match count's
ratchet (ask: match-count-ratchet — claim: corpus-match-today is fresh evidence for it), the
undocumented refusal prefixes (ticket: clean-prefix-reconcile — claim: refusal-prefix-share
adds two families to it), and whether every documented limitation really has an executable
twin (ask: doc-twin-overstatement — exclusion: whole-relation-shapes measured the same gap
in `test_known_limitations.py` and states the partial truth rather than becoming a sixth
site for the overstatement).

**Four findings recorded and not acted on here**, because the fence for this task is one
document. Each is a decision or a defect that belongs outside it:

1. **The seed-1804 `DIVERGE_VALUE`** in claim: campaign-verdicts-today is a live parity
   defect — a NaN *sign* reaching a string, invisible to the canonical `repr` comparison —
   and wants an xfail-strict pin plus a ticket, per the engine-bug process.
2. **C1 is gated at `MARGINALIZE_FUZZ_N = 25` while its text says 1,500-2,000**
   (claim: kpi-pointers-resolve). Under `kpis.md:12` that is an unacknowledged trade-off on
   a *control*, and the remedy is the owner's, not an editor's: correct C1's text to the
   depth that actually runs, or raise the default and pay the runtime. Both are changes to
   the KPI set, so both route through ask: kpi-set-change.
3. **`unsupported: comparison on BOOLEAN` is in the engine and in no document**
   (ask: exclusion-ratification (2)) — the second-largest refusal class in the campaign,
   86 of 944. It is a bookkeeping bug by `known-limitations.md:284`'s own rule, and it is
   named here rather than fixed here because the fix edits `known-limitations.md`.
4. **A static-only `ORDER BY` with ties freezes a nondeterministic order**
   (exclusion: whole-relation-shapes). Only a *row limit* refuses; the source's own measured
   sentence — `ORDER BY` does not fix ties — was not carried into the refusal. Two builds of
   the same function can then disagree, which is goal: serving-without-skew's own failure
   mode in its build-to-build face. It wants the same treatment the row limit got: a
   measurement and then a refusal, or a written reason it is acceptable.

Nothing else this document measured is left unrouted.
