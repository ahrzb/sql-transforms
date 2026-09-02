# The confit goal

**What this document is.** What confit is *for*, how much of it we intend to serve,
what we deliberately do not serve, and how each of those is measured. It sits above the
oracle spec (`packages/confit/docs/oracle/`) and the engine and testing specs that follow
it: the oracle spec defines *correct*, this document says *how much correct, over which
queries, and why the rest is out*.

**Where the numbers are.** This document holds **definitions, methods and enforcement
pointers, and no dated reading**. Every measured value — acceptance rates, corpus and floor
counts, bench tables, refusal histograms — lives in a dated report under
`packages/confit/docs/reports/`; the first is `2026-09-02-goal-baseline.md`, and a later
reading is a **new dated file**, never an edit to this one. Facts are still measured or read
from a pin and never recalled; what changed is that a *reading* no longer lives in prose
here, which is how a bar goes a month unre-run and nobody notices. Current tasks, tickets
and defects are the implementation loop's, and the loop reports into those files.

**Non-circularity, the same rule the oracle spec runs on.** This document is the
authority. Code, tests, pins and gate floors are its **enforcement**, never its
definition. A goal is not "whatever the gates currently pass" — if it were, every
regression would redefine the goal the moment it landed. `Enforced-by:` names the thing
that makes a statement hold; `Verified-by:` names the test, pin, gate line or dated
measurement that would catch it not holding. A statement nothing checks says `Unverified`
and says so plainly.

**How an item is named.** Nothing is named by a number. Every item carries a **slug** —
a short kebab-case noun phrase naming its *subject*, so the name still reads true when
the ruling changes. Slugs are assigned once; renaming one is a tombstone line naming
both. Moving an item to a dated report is **not** a rename and not a retirement: it keeps
its slug there. The family is carried by the citation, not by the slug:

| kind | written as | example |
|---|---|---|
| goal | `goal: <slug>` | goal: two-outcome-contract |
| exclusion-ledger row | `exclusion: <slug>` | exclusion: whole-relation-shapes |
| KPI | `kpi: <slug>` | kpi: acceptance-rate |
| claim of fact | `claim: <slug>` | claim: model-surface-split |
| ASK block | `ask: <slug>` | ask: acceptance-target |

Slugs are unique across every family and across the oracle spec, and the form above is
used at the definition and at every reference, so `grep -rn "<slug>"` finds an item and
everything that cites it.

**Two markers keep the normative half honest.** `[PROPOSED]` — a statement this document
would like, which nobody has ruled on; it holds a slug only so a ticket can cite it.
`[FACT]` — a measured statement of current state with no decision attached. Every
decision that belongs to the owner is an ASK block or carries `[PROPOSED]`; none of them
is written as settled. That includes every acceptance-rate target, every choice of which
query class is next, ratification of the exclusion ledger, and any change to the KPI set —
the set itself is in force in section 5, which is a ruling and not a proposal.

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
`packages/sql-transform` (kpi: training-round-trip).
*Verified-by:* `packages/confit/README.md:11-22` (the contract and this argument);
kpi: training-round-trip and kpi: engine-parity (section 5.2).

**goal: two-outcome-contract.** For any SQL handed to `DuckDBInferFn`, exactly one of two
things happens: it serves bit-for-bit identical to the oracle, or it refuses at build with
a `ValueError` naming the construct. Nothing is approximated, silently dropped, or widened
at inference time. This is the load-bearing goal, and everything in section 2 follows from it.

**The third mode exists and is enumerated — that is the whole difference.** The absolute
above is "no *unenumerated* third mode", not "no third mode", and writing it the strong
way would make this document false on its own evidence. The enumeration is the oracle
spec's divergence ledger
(`packages/confit/docs/oracle/07-the-divergence-ledger.md`), whose rows *serve* with a
consciously different surface. **Which rows are in force, and at what severity, is the
ledger's status column to say, never this document's** (section 6) — so the ledger is
pointed at here and never copied. The pointer is the structural part: a contract that
admits enumerated exceptions is only honest while the enumeration has a single home.
*Enforced-by:* build-time refusal at `DuckDBInferFn(...)` construction throughout
`packages/confit/src/specializer/`; the ledger for the enumerated exceptions.
*Verified-by:* kpi: no-third-mode (section 5.2, the corpus's
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
front matter's rule against goals-defined-by-current-measurement as binding here: a
measured gap is evidence for the regime, and a regression that moved it would not fail
anything.
*Verified-by:* the serving bench, read into the dated baseline report. **No target number
is in force** — the budget is a regime, not a bound anyone has ruled on. See
ask: kpi-set-change.

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
never into a wrong answer. Section 2 is this goal's frame and its yardsticks.
*Enforced-by:* **two of the four yardsticks in section 2, and only two.** The dialect
floors gate ratchets its two constants (`test_dialect_corpus_gate.py:37`,
`test_dialect_cross_engine_gate.py:50`). The corpus match count is *printed and never
asserted* — `test_corpus_replay.py:180-184` prints it, `:186` asserts only `not fails` —
and the campaign acceptance rate has no gate at all. A count records a goal; it does not
make one hold. That gap is what ask: acceptance-target and kpi: ladder-ratchet are about.
*Verified-by:* kpi: coverage-ladder (section 5.3, "Progress = moving queries from
REFUSED to MARGINALIZED (never to FAILED)"); the four line reads above.

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
(`04-verdicts-agreement-abstention-refusal.md:27-29`), so a reading of a campaign says when
they were zero rather than leaving them out.

`REFUSED` is the shape of "not accepted". `UNSHIPPED` is **not**: an unshipped case
*builds and serves*, and only the comparison is withheld — which is why it sits inside the
acceptance numerator and is the divergence ledger's business, not the refusal
histogram's. The two retire differently all the same: a refusal retires by a decision to
serve the construct, an unshipped width by the width landing.

**Acceptance is defined as cases that built rather than refused**, and that numerator
deliberately contains the unshipped widths, the trap agreements and any live parity
finding — on this definition a parity defect counts as accepted, which is correct for a
*scope* metric and is exactly the reason acceptance can never stand in for parity.

### 2.1 The four yardsticks

What the accepted surface is read with. Each row is a **method**, not a number; the numbers
are the dated report's.

| yardstick | what it reads | what runs it | ratcheted? |
|---|---|---|---|
| mined-corpus replay | of the statements mined from DuckDB's own test suite, how many replay bit-exact, how many refuse cleanly, how many FAIL | `test_corpus_replay.py` | no — the match count is printed, only zero-FAIL is asserted |
| dialect L2 floor | how many mined statements survive parse-then-print invisibly to the oracle | `test_dialect_corpus_gate.py` | **yes**, an asserted floor |
| dialect L3 cross-engine floor | how many match through a second engine's dialect | `test_dialect_cross_engine_gate.py` | **yes**, an asserted floor — but only in an environment that has `pyspark`, and the fixture fails loudly rather than skipping |
| campaign verdict census | over a seed range of the generated grammar, the verdict distribution and the acceptance rate that falls out of it | `python -m fuzz.runner` (a manual CLI, **not** a standing gate) | no |

A fifth number is cited from across the package boundary: kpi: coverage-ladder (section 5.3)
pins the `sql_transform` admission ladder (mined marginalized/refused, and the curated
three-way split). That ladder is `sql_transform`'s, not confit's (goal: engine-half-only);
it is cited because acceptance growth is measured on both sides of the boundary, and it is
the one ladder whose *split* has a pin test that fails when the number moves. It takes two
tests to say so — `_corpus_test.py::test_progression_totals` pins the totals and
`::test_mined_corpus_scoreboard` pins the split — so citing the totals test alone would name
a gate that cannot catch a drift in the split.

**Campaign-validity caveat, and it bounds every acceptance number this document's methods
produce.** Those rates are over the **generated grammar**, which is not query space. The
oracle spec records the caveats directly: the campaign fuzzer is a manual CLI and *not* a
standing gate — only the regexp fuzzer is (claim: regexp-fuzz-gate's correction); there is
no coverage denominator that means anything yet (claim: coverage-denominator, `[PROPOSED]`);
a rising abstention rate would read as generator drift and is not reported
(claim: abstention-rate, `[PROPOSED]`); and the mined corpus's expected rows are an
optimizer-**on** recording with no provenance, so the mined ladder is measured against a
different reading of DuckDB than claim: oracle-identity names (claim: zero-fails-gate's
second correction). A share of a grammar is a real measurement of a synthetic population,
and nothing converts it into a statement about the SQL people write.

> ### ask: acceptance-target — is there an acceptance-rate target, and what is it?
>
> Neither the campaign acceptance rate nor the mined-corpus match count has a target, and
> neither has a ratchet. Three shapes, and the choice is yours:
>
> **(a) No target, dated reporting only.** Cheapest, and honest about the denominator
> problem: a grammar rate is not a query-space rate, so a target on it optimizes the
> generator as readily as the engine. Cost: nothing notices a slip, and an ungated ladder
> has already slipped unnoticed once.
>
> **(b) Ratchet, no target.** "Never decreases", the rule the dialect floors already run.
> Cost, and it is real: a deliberate scope *reduction* becomes a gate failure. The oracle
> spec's ask: match-count-ratchet is the same question for the mined corpus and should be
> answered with this one, not separately.
>
> **(c) A named target per surface**, e.g. "N% acceptance over the campaign grammar by the
> time decimal arithmetic ships". Cost: it needs the denominator work
> (claim: coverage-denominator) before the number means anything.
>
> My read: (b) for the corpus and dialect ladders, which have stable denominators, and (a)
> for the campaign rate until a denominator exists. Not applied — this is your call.
>
> *Context:* the current readings, and the slip, are in
> `packages/confit/docs/reports/2026-09-02-goal-baseline.md`.
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
> | decimal arithmetic | retires exclusion: unshipped-decimal-arithmetic entirely; empties the only `UNSHIPPED` bucket | an exact decimal lane through the expression tree, not just the join lane that already ships |
> | HUGEINT + the unsigned family (the i128 lane) | retires half of exclusion: wide-integer-lanes; the cranelift dependency was verified GO 2026-08-15 | i128 arithmetic and its overflow traps across both backends |
> | narrow-lane overflow traps | closes the one place a narrow lane serves an i64 value on the row path where `infer_arrow` refuses | trap threshold per declared width |
> | nested / struct-valued outputs | retires the whole-struct half of exclusion: non-scalar-values | output schema work at the Arrow boundary |
> | `USING` / `NATURAL` self-joins under `shape='many'` | closes a named follow-up inside exclusion: multiplicity-by-default | small; it is a named rejection, not a model gap |
> | lifting the one-join-per-query restriction under `'many'` | multi-join serving | multiplicity composition, the hardest of these |
>
> The campaign's refusal histogram says which of these the *generator* reaches, not which
> your users need, and the ranking it produces is not the ranking intuition gives — that is
> a fact about the grammar, not a demand signal. It also holds a large class that belongs to
> no exclusion row at all; see ask: exclusion-ratification (2).
>
> *Context:* the histogram is in
> `packages/confit/docs/reports/2026-09-02-goal-baseline.md`.
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

The ledger. Each row: **what** is excluded, **why**, the **condition** that retires it, and
**how it rings** when that condition is met — or an honest flag that nothing rings.

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
(`:6289-6299`). The last three are not small: `QUALIFY` and the modifier class each sit near
the top of the campaign's refusal histogram, and `abs(k) OVER ()` was a silently-dropped
modifier on master until it was made to refuse by name.
*Why:* scope-by-product-decision. Their output shape is not one-row-in / 0..N-out, so they
are not row-at-a-time feature transforms.
*Retires when:* nothing — none intended. The carve-out already exists and is not a
retirement: a **static-tables-only** query is evaluated once by DuckDB at build and frozen,
so aggregation and `ORDER BY` serve there — with row limits refused, because which rows
survive a limit is not a function of the query (measured: four different answers across
twelve connections). **The carve-out has a hole its source names:** only a *row limit*
refuses, and `ORDER BY` does not fix ties (a tie fed from a `GROUP BY` flipped in 20 runs,
`known-limitations.md:116-117`). So a static-only tie-producing `ORDER BY` builds and
freezes whichever order that build's DuckDB run happened to get — two builds of the same
function can disagree with each other, which is goal: serving-without-skew's failure in its
build-to-build face rather than its train-to-serve one. Nothing refuses it today.
*Rings:* nothing rings — there is no trigger. Lifting one breaks
`packages/confit/tests/test_known_limitations.py` — which is the executable twin of
`known-limitations.md`, **not of this row**. Its docstring says so (`:1-7`, "Section
numbers mirror the document"), and its whole-relation parameterization (`:98-117`) covers
nine of the constructs enumerated above and not the rest. The oracle spec's
claim: doc-twin-totality measured that gap across five other sites and
ask: doc-twin-overstatement is open on it; this document does not become a sixth site.
*Verified-by:* `packages/confit/docs/known-limitations.md:95-120`;
`packages/confit/tests/test_known_limitations.py:1-7, :98-117`.

**exclusion: per-row-general-work.** Non-constant regex patterns, replacement strings,
regex options and extract-group indexes; anything that would compile or bind per row.
*Why:* specialization-inherent, and the direct negation of goal: pack-time-only-work.
DuckDB compiles regexes per row; we compile at prepare.
*Retires when:* a decision is taken to abandon compile-once for this construct. Effectively
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
*Retires when:* **decimal arithmetic ships and the `UNSHIPPED` bucket is empty** — and the
bucket empties for two different reasons, so it is checked, not just read.
*Rings:* **yes, loudly, and this is the one exclusion with a real bell.** The campaign
classifies these cases `UNSHIPPED` instead of comparing them, `fuzz.oracle._type_delta`
carries exactly one arm (decimal-against-float64) which is deleted when the feature lands,
and the runner prints the bucket in a section of its own — an empty section means either the
feature shipped or the grammar stopped reaching it, and both are worth seeing.
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
*Retires when:* the **i128 lane** ships (its cranelift dependency was verified GO
2026-08-15) for HUGEINT/unsigned; the **narrow-lane trap phase** ships for the traps.
*Rings:* partially. `test_integer_widths.py::test_unserved_static_type_refuses_by_name`
(`:856-871`) parameterizes the unsigned widths (and `float32`) and asserts the refusal names
the column and the type — so shipping any unsigned *static* width turns that test red by
name, on the day it ships. `test_known_limitations.py:180-192` pins the uint64 static the
same way. What still rings nothing: **HUGEINT**, unsigned *row* columns, and the
narrow-lane trap phase.
*Verified-by:* `packages/confit/docs/known-limitations.md:122-136`, `:175-187`;
`packages/confit/tests/test_integer_widths.py:856-871`.

**exclusion: non-scalar-values.** List-typed columns when referenced (unreferenced they
cost nothing), the struct as a whole value (`SELECT a`), bracket field access (`a['i']`),
struct fields whose own types are non-scalar, the list-valued regex functions
(`regexp_extract_all`, `regexp_split_to_array`, the STRUCT form of `regexp_extract`),
`decimal256` statics, decimal row columns, and the BLOB overload DuckDB picks for a bare
`NULL` `repeat` string. Structs of scalars, deep field paths and struct-star **do** serve.
*Why:* scope-by-product-decision — the row-schema vocabulary is scalar, and a non-scalar
output has no place in a dict row.
*Retires when:* nested output support ships, for the struct half; the BLOB lane ships, for
BLOB. `decimal256` is upstream-blocked and retires on nothing we control — DuckDB itself
refuses it at arrow register.
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
*Retires when:* the dialect frontend's own parser replaces sqlparser, for the precedence
cases; a matching regex engine lands, for the RE2 cases. Neither is scheduled.
*Rings:* **yes** — the standing regexp differential fuzzer runs in the normal gate
(N=250, fixed seed), and a new divergence fails with its reproducing seed.
*Verified-by:* `packages/confit/docs/known-limitations.md:197-200`;
`packages/confit/tests/test_duckdb_regexp_fuzz.py:13-20, :35-37`;
`packages/confit/docs/oracle/` claim: regexp-fuzz-gate.

**exclusion: resource-ceilings.** Pad/repeat counts past a 1 GiB string-builder budget (a
literal count refuses at build; a data-driven count traps at runtime), the regex
program-size guard, and the Arrow batch ceiling.
*Why:* **resource** — no gigabyte allocations in a serving engine, by decision. This is
the ground whose whole point is that the accepted cost has a number attached.
*Retires when:* a decision raises the number, which is a review, not a feature.
*Rings:* **for the string budget only.**
`known_divergences/test_string_budget.py::test_a_budget_breaking_literal_count_refuses`
(`:145-146`) asserts the refusal with `pytest.raises(ValueError, match="builder|GiB")`, so
that ceiling rings on a *change*. The **Arrow batch ceiling rings nothing**: no test touches
it, and its only mention in `test_arrow_boundary.py` is a prose comment (`:33-35`). A
`Verified-by` pointing at a comment names nothing that would fail, which is this document's
own definition of not-verified — so: **`Unverified` for the Arrow half.** This row is a live
instance of ask: exclusion-ratification (1), inherited from the same pointer in
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
*Retires when:* nothing — none intended. Decided 2026-08-17.
*Rings:* **yes** — the campaign reads DuckDB twice per case and labels these `DIVERGE_OPT`,
so the class is counted rather than absorbed. It is a **reported finding, not an accepted
class** (the oracle spec's claim: contract-surface-gap), so its standing count is a cost the
report carries, never a bucket that quietly grows.
*Verified-by:* `packages/confit/docs/known-limitations.md:20-39`, `:231-257`;
`packages/confit/docs/oracle/` claim: oracle-identity, claim: optimizer-bracket.

**exclusion: statistics-dependent-kernels.** Behaviors that depend on column *statistics*
— ILIKE's NUL handling selects a different kernel depending on sibling rows — are excluded
from the corpus by name, and the engine takes the ASCII-kernel (NUL-transparent) behavior.
*Why:* specialization-inherent. A row-at-a-time engine cannot reproduce statistics-
dependent semantics even in principle.
*Retires when:* nothing. This one is permanent by construction.
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
*Retires when:* the one-join-per-query restriction lifts, and `USING`/`NATURAL` self-joins
land — both named follow-ups, neither scheduled.
*Rings:* nothing rings for the retirement; the shape contract itself is pinned in
`packages/confit/tests/test_shape_contract.py`.
*Verified-by:* `packages/confit/docs/known-limitations.md:76-93`.

> ### ask: exclusion-ratification — does this ledger become the exclusion list?
>
> Ten rows above, all `[PROPOSED]`, assembled from `known-limitations.md`, the engine's
> refusal sites and the campaign's refusal histogram. Ratifying them means three things
> bind: the **ground** assigned to each (specialization-inherent vs
> scope-by-product-decision vs resource, which is what makes "we refuse" auditable), the
> **condition** that retires each, and how many rings nothing.
>
> Three specific pieces I would like ruled on rather than assumed:
>
> **(1) Are the silent rows acceptable? Counted from the `Rings:` lines, it is seven of
> ten.** exclusion: whole-relation-shapes, per-row-general-work, non-scalar-values and
> statistics-dependent-kernels ring nothing at all; exclusion: multiplicity-by-default rings
> nothing *for its retirement*; exclusion: resource-ceilings rings on a change and not on a
> retirement, and its **Arrow half rings nothing at all**; and exclusion: wide-integer-lanes
> rings for its unsigned-static half and not for the rest. That count moved in both
> directions when it was re-measured, which is itself the argument for ratifying an
> inventory rather than inheriting one. Silence is fine where the condition is "never"; it
> is a live gap for exclusion: non-scalar-values and exclusion: multiplicity-by-default,
> whose conditions are real scheduled-ish work. The pattern that *does* work is
> exclusion: unshipped-decimal-arithmetic's: a classified verdict, a single code arm deleted
> on landing, and a report section that empties.
>
> **(2) Is the enumeration complete? Measurably not.** The campaign's second-largest refusal
> class appears in **no** exclusion row and **nowhere** in `known-limitations.md`, and the
> modifier class reached the ledger only by being added to exclusion: whole-relation-shapes
> in the pass that wrote this document. By `known-limitations.md:284`'s own rule a refusal in
> code and in no document is a bookkeeping bug to file. Behind that: the engine's
> `PrepareError` refusal sites outnumber these ten rows by two orders of magnitude, and **no
> mechanism maps a refusal site to a ledger row**. Ratification either accepts that the
> ledger is a curated summary rather than a cover, or asks for the mapping.
>
> **(3) Do the three grounds cover what actually refuses?** They classify *scope* decisions,
> and the histogram holds a class that is not one: a UDF whose declared return shape is
> wrong (`src/duckdb/mod.rs:606-609`) is neither specialization-inherent, nor
> scope-by-product-decision, nor resource — the **caller declared their UDF wrong**. Adding a
> message prefix would not classify it, because the taxonomy has no slot for
> caller-declaration errors. Such refusals nevertheless sit in the same `REFUSED` bucket that
> feeds the acceptance rate. Either a fourth ground, or a rule that they are excluded from
> the acceptance denominator — both are your call, and the second is the one
> kpi: acceptance-rate depends on.
>
> *Context:* the class sizes and the refusal-site census are in
> `packages/confit/docs/reports/2026-09-02-goal-baseline.md`.
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
*Verified-by:* kpi: engine-parity's `Enforced-by:` line (section 5.2), which names
`test_udfs.py` (`udf_check`) among its enforcing suites — its "Which DuckDB" line names no
suite; `test_udfs.py::test_a_udf_may_not_take_a_builtin_name`.

**claim: sklearn-is-the-reference.** On the surface DuckDB cannot run, an independent
sklearn reference plays the role optimizer-off DuckDB plays for SQL — and it is a
**reference**, not the oracle, which is why it comes with a named bound instead of bit
equality everywhere. The bound differs by family, and **there are three bounds in force,
not one** — naming only the loosest would be the quiet loosening section 5.1 forbids:

| surface | bound | where |
|---|---|---|
| tree scoring | **bit-exact** against `sklearn.predict` | `_trees_test.py::test_matches_sklearn_bit_exactly` |
| bare `StandardScaler` transformer columns | **`rtol=1e-12`** | `_transformers_test.py:67, :81, :204, :271` |
| `Pipeline([StandardScaler, PCA])` columns | **`rtol=1e-9`** | `_transformers_test.py:109, :127` |
| the campaign's sklearn metamorphic leg | **absolute `1e-9`**, not a relative tolerance | `fuzz/oracle.py:845` — `abs(o - p) > 1e-9` |

Four of the six `assert_allclose` sites in the transformer file are the tighter `1e-12`;
the campaign's `1e-9` shares a numeral with the transformer bound and not a meaning.
*Enforced-by:* the four rows above;
`packages/sql-transform/sql_transform/_transformers_test.py::_reference` (`:34`) is the
clone-per-group reference all the transformer rows compare against.
*Verified-by:* kpi: transformer-parity (section 5.2, the control this is — note its own
text names **no** tolerance, so the bounds above are read from the tests, not from the
KPI); `packages/confit/docs/oracle/` claim: metamorphic-self-legs (the `1e-9` leg, itself
recorded as having **no test of its own** — `Unverified`).
*Note, and it belongs to the owner rather than to this document:* kpi: transformer-parity's
extension for native transform families ("bit-exact for scaler/tree tiers, within the
declared per-family ulp bound for matvec tiers") is written against work that has not
landed. It is carried there as a pointer, not adopted.

---

## 5. Measurement and KPIs

**The KPI set is in force here.** It lived in `packages/confit/docs/kpis.md` until the
owner ruled that this document owns it (ask: kpis-absorb-or-defer, section 6); that file is
deleted and its definitions, enforcing-suite pointers and standing law are below, under
slugs. What did **not** move is its dated readings — bench tables and ladder counts are a
report's under the front matter's rule, and the current ones are in
`packages/confit/docs/reports/2026-09-02-goal-baseline.md`.

Companion: `packages/confit/docs/properties.md` — KPIs measure; properties state what must
remain true.

**The old codes, kept as pointers**, because merged PRs, backlog tickets, drafts and the
oracle spec cite them: C1 = kpi: training-round-trip, C2 = kpi: engine-parity,
C3 = kpi: binding-parity, C4 = kpi: transformer-parity, C5 = kpi: no-third-mode,
D1 = kpi: coverage-ladder, D2 = kpi: serving-latency. The slug is the name; the code is a
pointer and nothing is named by it.

### 5.1 Two kinds, and the standing law

Two kinds, optimized in opposite directions:

- **Control KPIs** are invariants. Their target is a fixed point; they are never
  "improved", only *defended* — adversarial work (fuzz, differentials, refusal pins) exists
  to hunt counterexamples. A control that can only be held at 99% is not a control, it is an
  unacknowledged trade-off.
- **Drive KPIs** are the ones actually optimized: coverage up, latency down. A drive gain
  only counts if it lands inside every control — the pins make the difference impossible to
  smuggle past.

**Loosening a control is a design decision, never a fix.** The only legitimate way a
control moves: explicitly, in a spec/draft, with the new bound named (precedent: matvec-tier
parity got a *declared* per-family ulp tolerance in DRAFT-23 — through review, not through a
failing test). And the rule that governs every trade below:

> Never trade a control for a drive gain. If a bar seems in the way, the move is a
> written, named tolerance — or a refusal.

**Three of the six candidates in 5.5 are bars, not drives, and one of them lands on
kpi: no-third-mode.** Saying otherwise here would be an unforced error in the one paragraph
whose job is to show the standing law is respected, so it is said plainly instead:
kpi: named-refusal-share is proposed as a **control at 100%** over refusal message quality,
and kpi: no-third-mode's own text is "refuses at construction with an error naming the
construct" — so it is a second bar over that clause, over a property that demonstrably does
not hold at 100% today, and the two-kind rule above calls a control adopted below its own bar
"not a control, it is an unacknowledged trade-off". kpi: ladder-ratchet is a never-decreases
rule and kpi: bench-refresh-cadence a staleness bound; neither is a drive. **None of the six
may be adopted in a form that weakens the seven in force**, and the sequence that respects
the law for the one that touches kpi: no-third-mode is spelled out in its own entry: drive
first, control only after the gap is closed. Nothing in 5.5 is adopted; ask: kpi-set-change
is the door.

### 5.2 The controls in force (5)

**kpi: training-round-trip.** `fit(train)` + serving, applied to the training set, is
**bit-exact** equal to running the original SQL with `__THIS__` = train (both at
`SET threads = 1` — DuckDB's parallel window aggregation is not bit-deterministic for
floats). Transformer columns are exempt from the *DuckDB* oracle (DuckDB cannot run them)
and gate against kpi: transformer-parity instead.
*Enforced-by:* `packages/sql-transform/sql_transform/_projection_test.py` (`gate()` across
the admitted surface, plus the seeded differential fuzz — `MARGINALIZE_FUZZ_N`, seed
20260729; 1,500-2,000-case runs at each widening loop).

**kpi: engine-parity.** Confit serves **bit-for-bit identical to DuckDB** — with the same
declared UDFs registered via `create_function`, when there are any — **or refuses at build
with a named error**. No third behavior.
*Which DuckDB:* the optimizer-off reading (`PRAGMA disable_optimizer`), decided 2026-08-17.
The ground and the user-visible cost are exclusion: optimizer-on-answers; the oracle spec's
claim: oracle-identity is the authority on the oracle's identity, and this entry does not
restate it.
*Enforced-by:* `packages/confit/tests/test_duckdb_*.py` (the wave suites),
`test_params_joins.py`, `test_udfs.py` (`udf_check` — the parameterized form of the
contract), `packages/confit/docs/known-limitations.md` (each refusal has an executable
twin). Internal sub-invariant: cranelift ≡ interpreter, byte-for-byte —
`packages/confit/src/specializer/exec/tests.rs` (500-seed random-IR differential; shared
helper code is the structural argument).

**kpi: binding-parity.** `infer` / `infer_batch` (Confit row path) equals `transform`
(DuckDB batch path) **value-for-value** on the same fitted artifact — one artifact
(serving_sql + params tables + UDF objects), two bindings, no divergence.
*Enforced-by:* `packages/sql-transform/sql_transform/_serving_test.py` (`serve_gate()`
across aggregates, transformers width-1/width-k, author UDFs, chains, unseen-group NULLs;
dict rows ≡ model rows).

**kpi: transformer-parity.** Transformer columns equal an **independent clone-per-group
sklearn reference** (fit and apply re-derived from scratch in the test, not via the library
code under test). The entry names no tolerance of its own; the three bounds actually in
force are read from the tests, in claim: sklearn-is-the-reference.
*Enforced-by:* `packages/sql-transform/sql_transform/_transformers_test.py`
(`_reference()`).
*Extension, recorded as written and not in force* (DRAFT-23, when native families land): a
native entry equals its `PythonTransform` fallback twin — bit-exact for scaler/tree tiers,
within the *declared* per-family ulp bound for matvec tiers. The gate is swap-the-entry:
same SQL, same statics, different udfs-list entry. It is written against work that has not
landed, so it is a pointer, not an adopted bound.

**kpi: no-third-mode.** Every query either **serves** (under the four controls above) or
**refuses at construction with an error naming the construct**. Silent wrongness — a query
that builds and quietly computes something else — is the one unrecoverable state; the
corpus's FAILED bucket is pinned empty.
*Enforced-by:* `packages/sql-transform/sql_transform/_corpus_test.py` (three outcomes:
MARGINALIZED / REFUSED / FAILED-must-be-empty), the refusal tables in every test module,
`packages/confit/docs/known-limitations.md`.

### 5.3 The drives in force (2)

**kpi: coverage-ladder.** How much of projection-SQL the marginalizer admits: the mined
scoreboard (queries lifted verbatim from DuckDB's own window test suite, with provenance)
and the curated corpus's three-way split. Pinned in
`packages/sql-transform/sql_transform/_corpus_test.py::test_progression_totals` ("the
metric, in one place — edit these pins when a loop widens support") for the totals, and in
`::test_mined_corpus_scoreboard` for the mined split — it takes both tests, so citing the
totals alone would name a gate that cannot catch a drift in the split.
Progress = moving queries from REFUSED to MARGINALIZED (never to FAILED) and growing the
corpus. Known headroom, roughly in order of value: step semantics for order-keyed windows
off the training support (DRAFT-21), static-table joins + frozen composition, IN-subqueries
as fitted sets, star bundles into transformers, typed takes (string features).
*Method for widening it:* add the query to the corpus first, watch it refuse, then implement
until it marginalizes — and extend kpi: training-round-trip's gate to the new family in the
same loop. Update the pins deliberately.
*Scope, stated because it is not confit's:* this ladder is `sql_transform`'s
(goal: engine-half-only); §2.1 says why it is cited from here.
*Current reading:* `packages/confit/docs/reports/2026-09-02-goal-baseline.md` §6.

**kpi: serving-latency.** Row-at-a-time serving cost on the wide-table scenarios in
`benchmarks/`. Two harnesses: `bench_serving.py` (pure-SQL path) and `bench_transforms.py`
(the transformer/UDF path). Absolute numbers drift with machine load between runs —
**compare WITHIN a run**; the honest cross-run metric is a ratio to a baseline row measured
in the same run, which the transformer path has (a 2-field query against a 1-field one) and
the pure-SQL table does not.
*The finding that sets priorities* (measured 2026-08-04): the extern/UDF machinery is cheap
and our own marshalling is ~400ns, while **~93% of a fitted transformer's per-row cost is
sklearn's own `transform()`** — 60,900ns for `StandardScaler.transform` on one row against
1,500ns for the identical arithmetic in numpy and 1,000ns in pure Python. sklearn's per-call
validation, not the boundary, is the bottleneck. Levers, in that measurement's order:

1. **Native UDF families (DRAFT-23)** — replaces the 60µs sklearn call with ~1µs of
   arithmetic. A ~100x lever on transformer queries and by far the dominant one.
2. ~~Single-evaluation field access~~ — **DONE**: k addressed fields share ONE
   `transform()` call per row on both paths, counted rather than timed
   (`_single_eval_test.py` asserts the call count; DuckDB merges the identical pure calls by
   CSE, confit reads k lanes off one ecall).
3. Vectorized `apply_batch` for `infer_arrow`; marshaller work as measured.

*Method for moving it:* measure first (`benchmarks/`), then swap entries behind the extern
slots — the five controls are the safety net; if an optimization needs a control loosened,
that is a draft + review, not a code change.
*No target number is in force* — goal: request-latency-budget is a regime, `Unverified` by
construction, and no gate, floor or pin bounds serving latency.
*Current reading:* `packages/confit/docs/reports/2026-09-02-goal-baseline.md` §7, which also
carries finding: bench-baseline-flip. The 2026-08-04 bench tables this entry used to carry
were dated readings and did not move; they are in git history at
`packages/confit/docs/kpis.md`.

### 5.4 What enforces them

**claim: kpi-pointers-resolve.** Every `Enforced-by:` pointer under the five controls in 5.2
resolves — the check itself is a dated reading and lives in
`packages/confit/docs/reports/2026-09-02-goal-baseline.md` §8, which carries the map suite by
suite.

**Four of the five controls, and the coverage-ladder drive, are enforced in the other
package.** Only kpi: engine-parity is confit's own (`_projection_test.py`,
`_serving_test.py`, `_transformers_test.py` and `_corpus_test.py` all live in
`packages/sql-transform`). Absorbing the set did not move a gate: this document's §1 says it
is about confit's half (goal: engine-half-only), and it now holds four bars its own package
does not enforce. That is stated rather than hidden — it was the strongest argument against
absorbing, and the ruling in section 6 took it knowingly.

**A pointer that resolves is not a bar that holds at its written depth.** Where a control's
text names a run depth and its gate reads that depth from an environment variable, the
control holds at the default, not at the text — under 5.1 that is an unacknowledged
trade-off, and the remedy is a decision (correct the text, or raise the default and pay the
runtime), not an edit. Which control that is today, and by how much, is the dated report's;
the decision routes through ask: kpi-set-change.

### 5.5 Proposed KPI candidates — none adopted

Six candidates, **separate from the seven in force above**. Each names the measurement that
would back it and what it costs. **None is adopted** — changing the set is
ask: kpi-set-change. Definitions and methods only: no candidate carries its current value,
which lives in the dated report.

**kpi: acceptance-rate.** **[PROPOSED]** A **drive**: the fraction of generated-grammar
cases that build rather than refuse, reported per campaign with its seed range.
*Measurement that backs it:* exists and runs today — `python -m fuzz.runner` over a named
seed range, deterministic and re-runnable; acceptance is the complement of the `REFUSED`
count over the case count.
*Cost:* the denominator is a grammar, not query space, so the number can be moved by
editing the generator — and that is not hypothetical. `fuzz/gen.py:1060` reads
`w = rng.choice([1, 2, 3])  # width-1 list must REFUSE`: the generator deliberately emits
UDF return shapes it knows are rejected, and changing that one `rng.choice` moves acceptance
by percentage points with no engine change at all. Adopting it as a *drive* without
claim: coverage-denominator's triple-axis work invites optimizing the wrong thing; adopting
it as a *reported statistic* costs nothing. Either way ask: exclusion-ratification (3) has
to settle first whether a caller-declaration error belongs in the denominator.

**kpi: findings-per-campaign.** **[PROPOSED]** An explicit **zero-control**: findings per
N seeds, counted **beside** `DIVERGE_OPT` rather than with it. The wording matters: the
oracle spec's claim: contract-surface-gap is explicit that `DIVERGE_OPT` "stays a reported
finding ... rather than an accepted class", so this KPI may hold it out of the *bar* — a
zero-control over a class we knowingly tolerate would be red on day one — but never out of
the *report*. A control also has to name `TIMEOUT`/`PANIC`, or a loaded machine reads as a
finding.
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
its class list, driven to empty.
*Measurement that backs it:* the runner already prints the bucket in its own section, and
`_type_delta` carries one arm per unshipped feature — so the bucket is exactly the ledger's
open widths, already machine-readable.
*Cost:* near zero; this is the one candidate whose machinery is already built and whose
retirement bell already rings (exclusion: unshipped-decimal-arithmetic). Note the empty
state is ambiguous — the runner's own comment says an empty section means either the
feature shipped **or the grammar stopped reaching it**, so burn-down to zero needs the
generator checked, not just the number.

**kpi: named-refusal-share.** **[PROPOSED]** A **control** at 100% — and it is the one
candidate that is a *bar*, over kpi: no-third-mode's own clause; 5.1 says so rather than pretending
otherwise. The property: every refusal carries a documented, actionable prefix **and names
the construct**.
*Measurement that backs it:* a script over the campaign's own verdicts measures the
**prefix half** — a refusal message either starts with one of the documented prefixes or it
does not — so any reading of it is a **floor** on this KPI, never a reading of it: the
frontend's catch-all `expression: {other}` site passes the prefix check while naming
nothing. The **construct-naming half has no measurement today** and would need one before
the bar could be set.
*Cost:* a control adopted while the property demonstrably does not hold is, by 5.1's own
definition, "not a control, it is an unacknowledged trade-off" — so adopting it means
**first** closing the undocumented refusals, which is the oracle spec's open
prefix-reconciliation work, and only then declaring the bar. Adopting it as a drive first,
then promoting it, is the sequence that respects the standing law. This is the candidate I
would rank first: it is the only one that measures whether goal: two-outcome-contract's
*second* outcome is actually usable, and nothing measures that today.

**kpi: ladder-ratchet.** **[PROPOSED]** kpi: coverage-ladder, extended: the corpus match
count, the dialect L2 and L3 floors, and the sql-transform admission ladder, each with a
never-decreases rule and a single dated home.
*Measurement that backs it:* all four already exist; two already ratchet (the dialect
floors), two do not.
*Two scope notes before this one is adopted, because both change what it costs.* The
sql-transform ladder is **not confit's** (goal: engine-half-only, and §2.1 says so in the
same breath as citing it) — a confit document proposing a gate over the other package's
admission ladder is annexation unless the owner rules the KPI cross-package. And the L3
Spark floor **cannot be read in an environment without `pyspark`**, where the fixture fails
loudly rather than skipping, so ratcheting it means naming the environment that checks it.
*Cost:* a deliberate scope reduction becomes a gate failure. That cost is not theoretical —
an ungated count has already slipped unnoticed, which is the argument *for*; and a scope
reduction is sometimes right, which is the argument *against*. Same question as the oracle
spec's ask: match-count-ratchet; answer them together.

**kpi: bench-refresh-cadence.** **[PROPOSED]** kpi: serving-latency carries a **staleness bound**: the
serving bench is re-run and re-recorded at a named cadence (per release, or per N commits),
and a number older than the bound is marked stale rather than quoted.
*Measurement that backs it:* the bench runs in a few minutes with a parity gate of its own,
so the cadence is affordable.
*Cost:* absolute numbers drift with machine load — the same cell has read a 57% spread
across two runs on one machine on one day — so a cadence produces noise unless what is
recorded is the **ratio** to a baseline row in the same run, which is what
kpi: serving-latency already has for the transformer path (a 2-field query against a
1-field one) and does *not* have for the pure-SQL table. Adopting this means picking
which ratio is the metric, and settling
`finding: bench-baseline-flip` in
`packages/confit/docs/reports/2026-09-02-goal-baseline.md` first — a cadence on a metric
whose baseline changed identity would re-record the confusion.

> ### ask: kpi-set-change — adopt any of the six, and as what?
>
> The KPI set is five controls and two drives, and changing it is yours alone. My ranking,
> with the reason, not as a recommendation to be rubber-stamped:
>
> 1. **kpi: named-refusal-share, as a drive now and a control later.** Nothing today
>    measures whether a refusal is usable, and a refusal that does not name its construct
>    fails goal: two-outcome-contract's promise as surely as a wrong value does. Adopting it
>    as a control at today's share would violate the standing law on its first day.
> 2. **kpi: unshipped-burndown, as a drive.** Its machinery already exists and already
>    rings; adopting it costs a line in 5.3.
> 3. **kpi: ladder-ratchet**, decided together with the oracle spec's
>    ask: match-count-ratchet, because they are one question.
> 4. **kpi: bench-refresh-cadence**, once the serving bench's baseline question is settled —
>    a cadence on a metric whose baseline changed identity would just re-record the
>    confusion.
> 5. **kpi: acceptance-rate**, as a *reported statistic* rather than a KPI, until a
>    denominator exists (claim: coverage-denominator).
> 6. **kpi: findings-per-campaign** last, and only at release cadence — a zero-control on a
>    fuzzer punishes the fuzzer for working.
>
> This ask also carries one decision that is not an adoption: where a control's written run
> depth and its enforced default disagree, **correct the text or raise the default**. Both
> are changes to the KPI set, so both land here.
>
> Adopting none is a legitimate answer and leaves the seven in force exactly as they are.
>
> *Context:* the readings behind all six, and the depth gap, are in
> `packages/confit/docs/reports/2026-09-02-goal-baseline.md`.
>
> *Binds:* all six `kpi:` slugs, and section 5's two-kind structure.

---

## 6. Where this document sits

Above the specs, below the owner. The intended shape of the set:

| document | answers |
|---|---|
| **this document** (`docs/goal.md`) | what confit is for, how much of it we intend to serve, what we exclude, how it is measured |
| `docs/reports/<date>-goal-baseline.md` | what those yardsticks read on a date — one file per reading, never an edit to an older one |
| `docs/oracle/` (merged) | what *correct* means — the oracle's identity, the verdict taxonomy, the comparison contract, pins, the divergence ledger |
| the engine spec (upcoming) | how the engine achieves it — lanes, slots, the specializer, the backends |
| the testing spec (upcoming) | how it is checked — gates, corpora, campaigns, what each suite is for |

The direction of citation runs downward: this document may cite an oracle-spec claim as
evidence for a goal, and the oracle spec does not cite goals. Where the two overlap the
oracle spec wins on *correctness* questions and this document wins on *scope* questions.
Concretely, and it is tested by two places above: section 1 **points at** the divergence
ledger as the enumerated exception to the two-outcome contract and takes the rows and their
status from the ledger's own column rather than copying them; and
exclusion: unshipped-decimal-arithmetic **names** the two rows it overlaps and leaves them
`unruled` where the ledger leaves them. Citing a row is deference; restating its verdict
would be re-litigation, and section 2 does not redefine a verdict either. The same
direction holds downward into the reports: a report reads this document's yardsticks and
never amends one.

**ask: kpis-absorb-or-defer — RULED: this document owns the KPIs.** Three options were on
the table: absorb, defer-and-split-by-kind, defer wholly. The owner ruled **absorb**.
`packages/confit/docs/kpis.md` is deleted; its five controls, its two drives and its
standing law are section 5, under `kpi:` slugs, and its dated readings stayed out under the
front matter's rule — they are the dated report's.

Two costs were known before the ruling and are taken, not discovered. **Citations:**
line-anchored references into that file — from the oracle spec (claim: fit-serving-oracle),
`properties.md`, drafts and merged PRs — no longer resolve; every live one was repointed at
a slug in the same commit, and the historical ones (backlog tickets, dated reports and
decision files) were left as the records they are. **Scope:** four of the five controls and
kpi: coverage-ladder are enforced in `packages/sql-transform`, so a document whose §1 says
it is about confit's half (goal: engine-half-only) now holds four bars its own package does
not enforce. 5.4 states that rather than hiding it.

---

## ASK index

### Ruled

An ASK leaves the open table only by being answered. The ruling text lives at the point in
the document where it binds, next to what it created.

| ask | ruling | where it landed |
|---|---|---|
| **ask: kpis-absorb-or-defer** | **absorb** — this document owns the KPIs; `packages/confit/docs/kpis.md` is deleted, its definitions and standing law move here under `kpi:` slugs, its dated readings stay in the reports | section 6 (the ruling) and section 5 (the set itself: kpi: training-round-trip, kpi: engine-parity, kpi: binding-parity, kpi: transformer-parity, kpi: no-third-mode, kpi: coverage-ladder, kpi: serving-latency) |

### Open (4)

| ask | question | binds |
|---|---|---|
| ask: acceptance-target | is there an acceptance-rate target, and does the ladder ratchet? | goal: growing-accepted-surface, kpi: acceptance-rate, kpi: ladder-ratchet |
| ask: next-query-classes | which query classes are next, in what order? | goal: growing-accepted-surface, four `exclusion:` rows |
| ask: exclusion-ratification | does the ten-row ledger bind — silent conditions, the classes it misses, and the ground it has no slot for? | every `exclusion:` slug, kpi: acceptance-rate |
| ask: kpi-set-change | adopt any of the six proposed KPIs, and as what kind? | all six proposed `kpi:` slugs in 5.5 |

**Current readings live in `packages/confit/docs/reports/`**, one dated file per reading,
starting with `2026-09-02-goal-baseline.md`. Anything this document once carried as a dated
number is there under the same slug — including the serving bench's baseline question, which
is a **finding with options** for the implementation loop rather than a question the owner
must rule on, and the loop-level findings each reading turns up. Those files also carry the
reproduction commands and their environment preconditions, which is where a fresh checkout
starts.

Three questions this document deliberately does **not** ask, because they are already open
in the oracle spec and forking them would split the answer: the corpus match count's
ratchet (ask: match-count-ratchet), the undocumented refusal prefixes (the oracle spec's
claim: refusal-message-prefixes already records that the prefix set is not exhaustive and
routes the fix), and whether every documented limitation really has an executable twin
(ask: doc-twin-overstatement — exclusion: whole-relation-shapes states the partial truth
rather than becoming a sixth site for the overstatement).
