# Goal yardsticks, baseline reading (2026-09-02)

**What this is.** The first reading of the goal document's yardsticks — reading **N=1**.
`packages/confit/docs/goal.md` is the **target**: what confit is for, where its scope edge is
drawn, and *how each of those is measured*; it deliberately holds no number. This file is the
**measurement of how close today's engine is to that target**, on one dated run, in one named
environment. Every difference between the two is stated here as a **gap**. It is a
**reading, not a rule**: nothing here amends a goal, ratifies an exclusion or adopts a KPI.

**A later reading is a new dated file in this directory**, never an edit to this one. That
is the whole point of the split: a document that carries its own readings in prose is how a
bar goes a month unre-run and nobody notices — which is exactly what the latency-reading
section below measured.

**Slugs, and the one new family.** Items keep the slugs they had when they lived in
`goal.md`; only the family changes, and each entry records its old name once.
**`gap: <slug>` is a divergence of the *current state* from `goal.md`'s target** — this is
not the oracle spec's `divergence:` family, which is a divergence of the *engine's behaviour*
from the *oracle* on the accepted surface; a gap closes when the work lands, a `divergence:`
closes when the ledger rules on it. `goal:`, `exclusion:`, `kpi:` and `claim:` citations
resolve in `goal.md`, as do `ask:` citations except the one its ASK index records as living
here; `gap:` and `finding:` resolve here; `claim:`, `divergence:` and `ticket:` citations
without a local definition resolve in `packages/confit/docs/oracle/`. **Sections carry slugs
too:** every numbered heading here and in `goal.md` ends in a kebab-case anchor — `## 3. The
gap ledger {#gap-ledger}` — and every cross-reference names that slug ("the gap-ledger
section"), never the number, which is reading order only. A heading whose subject is already
a slugged item (`### 4.1 finding: seed-1804`) is cited by that item and takes no anchor.

---

## 1. Environment and reproduction {#environment-and-repro}

Every number below was produced here, or is marked as unreproducible here and says why.

| | |
|---|---|
| worktree | master `2ba96e5` |
| DuckDB | 1.5.5 |
| build | `uv run maturin develop --release` |
| machine | Windows 11, 12 cores |
| campaign flags | `--workers 8 --timeout 20` |

**The build is a precondition, not a detail.** A fresh checkout ships no compiled
extension, so every number in this file is unreproducible until that build completes — and
`benchmarks/bench_serving.py:31-35` records that the *wrong* build silently corrupts the
engine rows rather than failing. See gap: bench-baseline-flip (d): the harness's own
recommended command is `uv run --reinstall-package confit python -c pass`, which is not the
command this environment used.

| reading | command |
|---|---|
| corpus replay | `uv run pytest packages/confit/tests/test_corpus_replay.py -s` |
| dialect L2 floor | `uv run pytest packages/confit/tests/test_dialect_corpus_gate.py -s` |
| dialect L3 Spark floor | `uv run pytest packages/confit/tests/test_dialect_cross_engine_gate.py -s` — **needs `pyspark`** |
| campaign census | `python -m fuzz.runner --seed 0 --n 2000 --workers 8 --timeout 20` (from `packages/confit/`) |
| refusal histogram / prefix share | a ~40-line script over the campaign's `findings.jsonl` |
| serving bench | `uv run python -m benchmarks.bench_serving` |
| sql-transform ladder | `uv run pytest packages/sql-transform/sql_transform/_corpus_test.py` |

The campaign flags matter and are **not** the defaults: `fuzz/runner.py:213-214` defaults to
`--workers 4 --timeout 30`.

---

## 2. KPI status {#kpi-status}

One row per KPI in force (`goal.md`'s controls-in-force and drives-in-force sections define
them; this table reads them). All cells read **2026-09-02** in the environment-and-repro
section. "Pointer resolves" means the enforcing suite named in `goal.md` exists and runs,
which the enforcement-as-read section checked suite by suite; it is not a claim that the bar
was re-run here.

| kpi | defined at | reading, 2026-09-02 | read from |
|---|---|---|---|
| kpi: training-round-trip | `goal.md` controls-in-force | pointer resolves; the standing gate runs at depth **25**, not the 1,500-2,000 the definition names | enforcement-as-read, finding: c1-depth |
| kpi: engine-parity | `goal.md` controls-in-force | **one live counterexample**: seed 1804 of 2000, one `DIVERGE_VALUE`; corpus replay 0 FAIL of 678; the 7 `DIVERGE_OPT` are exclusion: optimizer-on-answers' standing cost, reported not accepted | acceptance-reading, finding: seed-1804 |
| kpi: binding-parity | `goal.md` controls-in-force | pointer resolves (`_serving_test.py::serve_gate`); no dedicated reading taken this run | enforcement-as-read |
| kpi: transformer-parity | `goal.md` controls-in-force | pointer resolves (`_transformers_test.py::_reference`); the three bounds in force were not re-read this run | enforcement-as-read |
| kpi: no-third-mode | `goal.md` controls-in-force | the FAILED bucket is **empty** (0 of 678 corpus, 0 of 2000 campaign); the "names the construct" half of the clause is **unmeasured** — 88.7% prefix presence is a floor, and 107 refusals carry no documented prefix | refusal-histogram, gap: undocumented-refusal-prefixes |
| kpi: coverage-ladder | `goal.md` drives-in-force | mined **11 marginalized / 11 refused of 22**; curated **39 / 17 / 5**; corpus replay **547 of 678** (three below the last number written down); dialect L2 **288/678**, exactly on its floor; L3 Spark floor 260 **unread** — no `pyspark` here | mined-statements, l2-and-l3-floors, authoring-side-ladder, gap: corpus-match-slip |
| kpi: serving-latency | `goal.md` drives-in-force | `spec` p50 **2,300-8,800 ns** per call at n=1 against **5.3-13.1 ms** for DuckDB-per-call — the regime holds; against the surviving Python twin row the engine is **1.13-2.05x slower** where the last written reading said 1.5-1.7x faster; the recorded table is 389 commits old | latency-reading, gap: bench-baseline-flip |

Two of the seven have a reading that is a **gap against the KPI's own text** rather than a
value: kpi: training-round-trip (depth) and kpi: no-third-mode (the unmeasured half). Both
are routed in the left-open section.

---

## 3. The gap ledger {#gap-ledger}

**A gap is a divergence of the current state from `goal.md`'s target.** Every entry names
what the goal targets, what the engine does today, the ground, the condition that closes it,
and its measured size where a histogram or campaign gives one. Five entries were rows of
`goal.md`'s scope-edge section until this reading: they were there because something is **not
built yet**, which is distance and not the target's edge, and `goal.md`'s scope-redirects
table records the redirect. The rest were found by measuring.

A gap is not a bug. Where the engine violates a control, that is a **finding** and lives in
the findings-not-gaps section — a control violation is never a gap to live with.

### 3.1 Gaps that were scope rows {#former-scope-rows}

**gap: unshipped-decimal-arithmetic.** *(was `exclusion: unshipped-decimal-arithmetic`.)*
*Target:* DECIMAL expressions serve bit-exact, and the `UNSHIPPED` bucket is empty.
*Today:* what is **missing** is narrower than the heading suggests, so it is worth stating
what already serves: comparisons, joins, `CAST(d AS DOUBLE)` and `SELECT *` over decimals all
serve (`known-limitations.md:148`), and decimal *statics* serve exactly in an i128 lane. What
refuses is **expressions** over DECIMAL — arithmetic, `CAST` to anything but `DOUBLE`, and
`COALESCE`/`CASE`/`greatest` unifying a decimal with a non-identical type.
**A DECIMAL literal does not refuse — it *serves*, at a different width.** Measured
2026-09-02, `SELECT 1.5 AS o0 FROM __THIS__` builds and returns `double` where the oracle
returns `decimal128(2,1)`; the case builds and serves, and the campaign classifies it rather
than comparing it (the bucket block below).
The visible consequence is that `CAST(-2.5 AS BIGINT)` on a bare literal
rounds half-to-even here and half-away-from-zero there. And the real split is not
literal-vs-static-column but **row-path vs static-only path**: the same literal reached
through a static-tables-only query never prepares, so DuckDB evaluates it and it comes back
`decimal128(2,1)` and `AGREE`
(`test_fuzz_smoke.py::test_the_static_only_leg_has_no_unshipped_width_to_classify`).
*Ground:* scope-by-product-decision, pending the lattice phase. Each of the refusing
expressions used to serve a silently wrong double, which is why they refuse now rather than
approximate — the refusal is correct behaviour under goal: two-outcome-contract; the *gap* is
that they refuse at all.
*Closes when:* **decimal arithmetic ships and the `UNSHIPPED` bucket is empty** — and the
bucket empties for two different reasons, so it is checked, not just read.
*Size:* **14 of 2000** campaign cases (0.7%), all class `decimals` (the acceptance-reading
section).
*Rings:* **yes, loudly, and this is the one entry with a real bell.** `fuzz.oracle._type_delta`
carries exactly one arm (decimal-against-float64) which is deleted when the feature lands,
and the runner prints the bucket in a section of its own — an empty section means either the
feature shipped or the grammar stopped reaching it, and both are worth seeing.
*Overlaps the divergence ledger:* the served-width fact is divergence: decimal-literal-typing
and the rounding fact is divergence: decimal-cast-rounding, both `unruled` there pending
ask: float-tolerance-list. The *scope* question is this entry's; the *symptom's* status is
theirs, and this entry does not settle it.
*Verified-by:* `packages/confit/docs/known-limitations.md:138-174`;
`packages/confit/fuzz/oracle.py:124-145` (the note), `:498-505` (the arm),
`packages/confit/fuzz/runner.py:168-176` (the report section);
`packages/confit/tests/test_fuzz_smoke.py::test_an_unshipped_lane_is_classified_and_never_value_compared`.

**The `UNSHIPPED` bucket, in one place.** The verdict itself is the oracle spec's
(claim: unshipped-verdict, `packages/confit/docs/oracle/05-the-comparison-contract.md:206`);
what it means for *this* gap is collected here, because it is measurement vocabulary and
`goal.md`'s acceptance-frame section deliberately carries none of it. **A case whose answer
exercises this gap's enumerated width is classified, never value-compared**: it counts as
neither agreement nor finding, and it never enters `findings.jsonl`. Four handlings were
available and three of them lie. Casting both sides and comparing buries a real width
difference inside `AGREE`. Grading it a divergence floods the findings with one known,
already-documented fact, and a findings file that is mostly known facts is a findings file
nobody reads. Not generating the case leaves the gap unmeasured and quietly shrinks the
denominator, so the gap stops costing anything on paper the moment it stops being visible. A
bucket of its own is the only handling that keeps both numbers honest: acceptance still counts
the case as built (it did build, and it did serve), and parity still refuses to claim anything
about a comparison that never happened. The bucket is also **self-retiring** — when decimal
arithmetic ships, the single `_type_delta` arm goes with it and the bucket must empty, so a
case still landing there
afterwards is a real bug rather than a known gap, which is why *Closes when* checks the
empty state instead of reading it. Today it holds exactly the 14 cases counted under *Size*,
all class `decimals`.

**gap: wide-integer-lanes.** *(was `exclusion: wide-integer-lanes`.)*
*Target:* every declared integer width serves, or refuses because we decided it should —
not because the lane does not exist.
*Today:* HUGEINT and the whole unsigned family refuse by name rather than collapse to i64;
`float32` base tables refuse; `float32` and unsigned *static* columns refuse rather than
widen. Narrow lanes type in DuckDB's lattice but compute in i64, so until the trap phase
lands an overflowing narrow lane serves the i64 value on the row path and refuses at the
`infer_arrow` boundary.
*Ground:* specialization-inherent for the lane widths as they stand (the engine computes in
i64/f64/str/bool), scope-by-product-decision for the refusal-instead-of-widen rule. The
catalogue used to widen `float32` and the unsigned widths and **both diverged silently**,
which is the measurement that produced the rule — and that rule is not the gap. The gap is
the missing lane.
*Closes when:* the **i128 lane** ships (its cranelift dependency was verified GO 2026-08-15)
for HUGEINT/unsigned; the **narrow-lane trap phase** ships for the traps.
*Size:* not separated in the 2026-09-02 histogram — the generated grammar does not reach the
wide widths often enough to rank (the refusal-histogram section).
*Rings:* partially. `test_integer_widths.py::test_unserved_static_type_refuses_by_name`
(`:856-871`) parameterizes the unsigned widths (and `float32`) and asserts the refusal names
the column and the type — so shipping any unsigned *static* width turns that test red by
name, on the day it ships. `test_known_limitations.py:180-192` pins the uint64 static the
same way. What still rings nothing: **HUGEINT**, unsigned *row* columns, and the
narrow-lane trap phase.
*Verified-by:* `packages/confit/docs/known-limitations.md:122-136`, `:175-187`;
`packages/confit/tests/test_integer_widths.py:856-871`.

**gap: non-scalar-values.** *(was `exclusion: non-scalar-values`.)*
*Target:* nested and struct-valued outputs serve; the row vocabulary stops being the reason
they cannot.
*Today:* refused — list-typed columns when referenced (unreferenced they cost nothing), the
struct as a whole value (`SELECT a`), bracket field access (`a['i']`), struct fields whose
own types are non-scalar, the list-valued regex functions (`regexp_extract_all`,
`regexp_split_to_array`, the STRUCT form of `regexp_extract`), `decimal256` statics, decimal
row columns, and the BLOB overload DuckDB picks for a bare `NULL` `repeat` string. Structs
of scalars, deep field paths and struct-star **do** serve.
*Ground:* scope-by-product-decision as written — the row-schema vocabulary is scalar, and a
non-scalar output has no place in a dict row. It is a gap rather than an edge because output
schema work at the Arrow boundary is the named way out, not a re-decision.
*Closes when:* nested output support ships, for the struct half; the BLOB lane ships, for
BLOB. **`decimal256` closes on nothing we control** — DuckDB itself refuses it at arrow
register, so that sub-case is upstream-blocked rather than unbuilt.
*Size:* unmeasured — the campaign's generator does not emit non-scalar outputs as a class.
*Rings:* nothing rings.
*Verified-by:* `packages/confit/docs/known-limitations.md:149-165`, `:207` — note the
`decimal256` and decimal-row-column facts this entry names are at `:149-150`, and `:166`
opens the DECIMAL-literals bullet, which belongs to gap: unshipped-decimal-arithmetic and
not here.

**gap: parse-divergence-guards.** *(was `exclusion: parse-divergence-guards`.)*
*Target:* these constructs serve, on a parser and a regex engine that agree with DuckDB's.
*Today:* they refuse — the `^` operator (it *is* pow in DuckDB, but sqlparser's precedence
differs, so mapping it computes a different tree silently), prefix `~`, `#`, `NOT GLOB`, and
the regex reject list: the RE2-vs-rust-regex differential battery's classes plus the twelve
the fuzzer found.
*Ground:* specialization-inherent **given today's dependencies** — these are the constructs
where a served answer would be *silently* different, which goal: two-outcome-contract forbids
outright. Refusing is right; needing to is the gap.
*Closes when:* the dialect frontend's own parser replaces sqlparser, for the precedence
cases; a matching regex engine lands, for the RE2 cases. Neither is scheduled.
*Size:* the reject list is the RE2 battery's classes plus twelve fuzzer-found patterns; the
standing differential re-swept to **zero new divergences over 40k cases across 8 seeds**
(the other-dated-readings section).
*Rings:* **yes** — the standing regexp differential fuzzer runs in the normal gate
(N=250, fixed seed), and a new divergence fails with its reproducing seed.
*Verified-by:* `packages/confit/docs/known-limitations.md:197-200`;
`packages/confit/tests/test_duckdb_regexp_fuzz.py:13-20, :35-37`;
`packages/confit/docs/oracle/` claim: regexp-fuzz-gate.

**gap: join-composition-limits.** *(was the second half of
`exclusion: multiplicity-by-default`; the first half — multiplicity only under an explicit
`shape='many'` — stayed in `goal.md`'s scope-edge section as a permanent decision.)*
*Target:* multi-join serving under `shape='many'`, and `USING`/`NATURAL` self-joins where
the shape allows them.
*Today:* one join per query under `'many'`; `USING`/`NATURAL` self-joins refuse under every
shape.
*Ground:* unbuilt. Multiplicity composition across two joins is the hard half; the
self-join case is a named rejection, not a model limit.
*Closes when:* the one-join-per-query restriction lifts, and `USING`/`NATURAL` self-joins
land — both named follow-ups, neither scheduled.
*Size:* unmeasured as a class; the generator's `shape='map'` WHERE-drop refusals (42 of 944,
the refusal-histogram section) belong to the permanent half, not to this entry.
*Rings:* nothing rings. The shape contract that *is* permanent is pinned in
`packages/confit/tests/test_shape_contract.py`; nothing pins these two follow-ups.
*Verified-by:* `packages/confit/docs/known-limitations.md:76-93`.

### 3.2 Gaps this reading measured {#measured-gaps}

**gap: undocumented-refusal-prefixes.**
*Target:* every refusal carries a documented, actionable prefix and names the construct —
goal: two-outcome-contract's second outcome, and what kpi: named-refusal-share would bar.
*Today:* **107 of 944 refusals (11.3%) carry no documented prefix**, in three families, and
two of the three are not named in the oracle spec's existing correction:
`shape='map': a WHERE clause can drop ...` (42) and its static-only twin (1);
`udf '<name>': a width-1 list return ...` (63); `static data mismatch: @0: duplicate ...` (1).
The construct-naming half is not measured at all.
*Ground:* bookkeeping — refusal sites that grew without a documented prefix, and no mechanism
mapping a site to a documented family.
*Closes when:* the oracle spec's ticket: clean-prefix-reconcile lands the missing prefixes,
and a measurement exists for the naming half.
*Size:* 107 of 944 (the refusal-histogram section). The corpus gate's narrower `_CLEAN` set
puts it differently again — 688 of 944, 72.9%.
*Verified-by:* claim: refusal-prefix-share (the refusal-histogram section);
`packages/confit/docs/oracle/` claim: refusal-message-prefixes.

**gap: undocumented-boolean-comparison.** *(was `finding: undocumented-boolean-comparison`;
it is a divergence from the target's enumeration, not an engine defect.)*
*Target:* every refusing construct is either an `exclusion:` row or a gap entry, with a
ground — `known-limitations.md:284`'s own rule is that a refusal in code and in no document
is a bookkeeping bug to file.
*Today:* `unsupported: comparison on BOOLEAN` (`specializer/frontend.rs:5322`) is the
**second-largest refusal class in the campaign** — 86 of 944, 9.1%, behind only `WITH` — and
it appears in **no** scope row, in no gap entry above, and **nowhere** in
`known-limitations.md` (`grep -c BOOLEAN` there returns 0). `unsupported: modifier on scalar
call abs` (`frontend.rs:6296`, 39 refusals) reached the ledger only by being added to
exclusion: whole-relation-shapes in the pass that wrote `goal.md`.
*Ground:* bookkeeping, plus the missing site-to-row mapping behind it (the refusal-histogram
section's refusal-site census).
*Closes when:* the class is documented and classified — as a scope decision in `goal.md`'s
scope-edge section if it is one, as a gap entry here if it is not.
*Size:* 86 of 944 (9.1%).

**gap: corpus-match-slip.**
*Target:* kpi: coverage-ladder is a **drive** — the mined-corpus match count grows, and never
silently shrinks.
*Today:* it shrank. **547** statements replay bit-exact where **550** is quoted at six
unhedged sites across *five* documents, and the oracle spec's ladder records 550 as a genuine
earlier reading (`53 -> 395 -> 505 -> 511 -> 529 -> 546 -> 550`), so the ungated match count
has moved **down by three** since it was last written down and nothing noticed.
*Ground:* the count is printed and never asserted (`test_corpus_replay.py:180-184` prints,
`:186` asserts only `not fails`), which is two of the four-yardsticks table's rows in
`goal.md` having no ratchet.
*Closes when:* the ratchet decision lands — this is fresh evidence for the oracle spec's open
ask: match-count-ratchet and for `goal.md`'s ask: acceptance-target, which should be answered
with it rather than separately — and the six stale sites are corrected.
*Size:* three statements, and six unhedged citation sites. One note on the cleanup's size,
because a ratchet decision turns partly on it: a bare `grep -rn 550 --include=*.md` on
2026-09-02 also finds the number in `oracle/09-version-bumps-and-mutability.md:56`, a second
occurrence in `reports/confit-architecture.md:30`, and
`docs/specs/2026-08-13-dialect-logical-plan-design.md:265` and `:298` — so the six *unhedged
current-state* sites the oracle spec's correction enumerates are a subset of the occurrences
a remediation would have to walk.
*Verified-by:* claim: corpus-match-today (the mined-statements section);
`packages/confit/docs/oracle/` claim: zero-fails-gate and its correction.

**gap: admission-ladder-headroom.** *(was kpi: coverage-ladder's "known headroom" list in
`goal.md`'s drives-in-force section.)*
*Target:* the `sql_transform` admission ladder keeps growing — progress is queries moving
from REFUSED to MARGINALIZED, never to FAILED.
*Today:* the named headroom, roughly in order of value: step semantics for order-keyed
windows off the training support, static-table joins plus frozen composition, IN-subqueries
as fitted sets, star bundles into transformers, typed takes (string features).
*Ground:* unbuilt authoring-side work. Note this ladder is `sql_transform`'s, not confit's
(goal: engine-half-only) — it is read here because acceptance growth is measured on both
sides of the boundary.
*Closes when:* each family lands by kpi: coverage-ladder's own method — corpus first, watch
it refuse, implement until it marginalizes, extend kpi: training-round-trip's gate in the
same loop.
*Size:* today's ladder is 11 marginalized / 11 refused of 22 mined, and 39 / 17 / 5 curated
(the authoring-side-ladder section).

**gap: native-transform-families.** *(carries kpi: serving-latency's 2026-08-04 priority
finding and kpi: transformer-parity's not-in-force extension, both of which were written in
`goal.md` against work that has not landed.)*
*Target:* serving cost is a single-digit-microsecond per-request number
(goal: request-latency-budget), and kpi: transformer-parity holds for every transform entry.
*Today:* **~93% of a fitted transformer's per-row cost is sklearn's own `transform()`** —
measured 2026-08-04: 60,900 ns for `StandardScaler.transform` on one row against 1,500 ns for
the identical arithmetic in numpy and 1,000 ns in pure Python, while the extern/UDF machinery
is cheap and our own marshalling is ~400 ns. sklearn's per-call validation, not the boundary,
is the bottleneck. Levers, in that measurement's order:

1. **Native UDF families** — replaces the 60 us sklearn call with ~1 us of arithmetic. A
   ~100x lever on transformer queries and by far the dominant one.
2. ~~Single-evaluation field access~~ — **DONE**: k addressed fields share ONE
   `transform()` call per row on both paths, counted rather than timed
   (`_single_eval_test.py` asserts the call count; DuckDB merges the identical pure calls by
   CSE, confit reads k lanes off one ecall).
3. Vectorized `apply_batch` for `infer_arrow`; marshaller work as measured.

*Ground:* unbuilt. The parity bound that would come with it is written and **not adopted**: a
native entry equals its `PythonTransform` fallback twin — bit-exact for scaler/tree tiers,
within the *declared* per-family ulp bound for matvec tiers, gated by swap-the-entry (same
SQL, same statics, different udfs-list entry).
*Closes when:* the native families land **and** that bound is adopted through review, which
under `goal.md`'s standing-law section is the only legitimate way a control moves.
*Size:* the 60,900 vs 1,500 ns split above; ~100x on the transformer path if the lever lands
as measured.

**gap: bench-baseline-flip.** *(was `finding: bench-baseline-flip`; it is a divergence from
kpi: serving-latency's own last recorded direction, with an unsettled cause.)*
*Target:* kpi: serving-latency is a **drive** — the number goes down, and the recorded
reading means something.
*Today:* the recorded table says the engine is 1.5-1.7x faster than the handwritten Python
twin. It is now **1.13-2.05x slower** than the twin row that still exists, on all five
scenarios — though only **four** of them have a 2026-08-04 baseline to have flipped from.
Four candidate readings, in the order they are cheapest to rule out:

**(a) Baseline changed identity.** The old `python` row returned pydantic models; the
surviving `python_dict` row returns plain dicts, which is cheaper. The magnitudes fit:
titanic's Python twin read 6,000 ns in the recorded table (`kpis.md:128`) and 2,200 ns today.
If that accounts for the whole delta, the old prose is simply obsolete and the fix is an edit.

**(b) Real regression.** 389 commits landed since the measurement, including the Arrow
schema API, the lane/slot seams and join-key work. Ruling this out costs one bisect against
`a6fa318` on the same machine.

**(c) Machine noise.** Other agents active. Not as cheap to dismiss as it looks:
`store_sales`' `spec` cell read 5,600 and 8,800 ns in two runs the same day.

**(d) A stale or debug wheel — and the harness warns about exactly this symptom, in
capitals.** `benchmarks/bench_serving.py:31-35`: "rebuild the wheel first (`uv run
--reinstall-package confit python -c pass`) — a stale wheel inflates ONLY the engine rows
and once produced a phantom 7x regression (caught by bisection, 2026-07-26)". Engine rows
uniformly slower with the Python row unchanged **is** the stale-wheel signature, it has
happened in this repo before, and it is the cheapest of the four to rule out. The
environment-and-repro section names `uv run maturin develop --release`, which is not the
harness's own recommended command; ruling (d) out costs one re-run after
`--reinstall-package`.

*Ground:* unknown, which is the gap — no bisect was run. What can be said: (c) alone is
unlikely to produce a uniform sign flip, (d) is untested and cheapest, and the answer changes
what goal: request-latency-budget is worth as a claim.
*Closes when:* (d) is ruled out by a re-run, then (b) by a bisect if it survives.
**Settle this before adopting kpi: bench-refresh-cadence** — a cadence on a metric whose
baseline changed identity would re-record the confusion.
*Size:* 1.13-2.05x slower on five scenarios; the recorded table is 389 commits old (the
latency-reading section).

---

## 4. Findings that are not gaps {#findings-not-gaps}

A **finding** is a defect or an enforcement fault: the engine, or a bar, failing on its own
terms. It is not distance from the target and it is never something to live with — a control
violation is a bug. Two of the three below are exactly that; the third is a bar enforced
below its written depth.

### 4.1 finding: seed-1804 — a live parity defect

The one `DIVERGE_VALUE` of 2000 seeds, seed 1804: `CAST(pow(-0.25e0, 0.1e0) AS VARCHAR)`
inside `struct_pack` gives `'nan'` on one side and `'-nan'` on the other — a NaN *sign*
reaching a string, which the canonical `repr` comparison elsewhere cannot see
(claim: blind-spots' NaN row). This is kpi: engine-parity's counterexample, not a gap:
parity on the accepted surface is a control fixed at 100%. Recorded, not fixed: per the
engine-bug process it wants an xfail-strict pin and a ticket.

### 4.2 finding: c1-depth — a control gated two orders of magnitude shallower than its text

kpi: training-round-trip says "1,500-2,000-case runs at each widening loop", but
`_projection_test.py:375` is `n = int(os.environ.get("MARGINALIZE_FUZZ_N", "25"))` — the deep
run is opt-in and the standing gate is 25. Under `goal.md`'s standing-law section a control
that only holds at its written depth when someone sets an environment variable is "an
unacknowledged trade-off", and the remedy is a decision, not an edit: **either correct the
text to 25, or raise the default and pay the runtime**. Both are changes to the KPI set, so
both route through `goal.md`'s ask: kpi-set-change.

### 4.3 finding: static-only-tie-order — a nondeterministic order can be frozen

`goal.md` exclusion: whole-relation-shapes states the target rule: inside the
static-tables-only carve-out, what a whole-relation construct selects is frozen **only when
it is a function of the query**, and both a row limit and a tie-producing `ORDER BY` fail
that test. Today only the *row limit* refuses. `ORDER BY` does not fix ties (a tie fed from a
`GROUP BY` flipped in 20 runs, `known-limitations.md:116-117`), so a static-only
tie-producing `ORDER BY` builds and freezes whichever order that build's DuckDB run happened
to get — two builds of the same function can disagree with each other, which is
goal: serving-without-skew's failure in its build-to-build face rather than its
train-to-serve one. **Nothing refuses it today**, which makes this a silent-wrongness class
and therefore a finding rather than a gap. It wants a measurement and then a refusal, or a
written reason it is acceptable.

---

## 5. Campaign verdicts and the acceptance reading {#acceptance-reading}

**claim: campaign-verdicts-today.** Over seeds 0-1999 of the generated grammar, measured
2026-09-02, and re-run twice with identical counts. **Corroboration, not independence:** the
second route was `fuzz.oracle.run_case_json`, which *is* what the runner's workers call
(`fuzz/worker.py` is nine lines around it), so the two runs share every line except the
subprocess and timeout wrapper. That wrapper is exactly the layer that can turn a verdict
into `TIMEOUT` under load, so it is the layer a second run most needs to exercise, and this
pair does not.

| verdict | count | share |
|---|---|---|
| `AGREE` | 1013 | 50.7% |
| `REFUSED` | 944 | 47.2% |
| `AGREE_TRAP` | 21 | 1.1% |
| `UNSHIPPED` | 14 | 0.7% |
| `DIVERGE_OPT` | 7 | 0.35% |
| `DIVERGE_VALUE` | 1 | 0.05% |

`TIMEOUT` and `PANIC` were **0** on this run, and that is a reading rather than a property:
a machine under load moves a case from its true verdict into `TIMEOUT`, which moves
acceptance by one. Any re-run that reports them non-zero has not found a defect, it has
found load — the fix is to re-run the named seed alone, not to re-record the table.

**Acceptance** — cases that **built rather than refused**, which is the only definition the
arithmetic supports — is **1056/2000 = 52.8%**. That numerator deliberately contains the 14
`UNSHIPPED` (they build and serve, only the comparison is withheld — the bucket's rules and
its rationale are gap: unshipped-decimal-arithmetic in the former-scope-rows section, which
is where all 14 sit), the 21 `AGREE_TRAP`, and the one live `DIVERGE_VALUE`: on this
definition the seed-1804 parity defect counts as accepted, which is correct for a *scope*
metric and is the reason acceptance can never stand in for parity. This paragraph is the
reading `goal.md`'s acceptance-frame section points at for what today's numerator contains; the
definition of acceptance stays there. All 7 `DIVERGE_OPT` are exclusion: optimizer-on-answers'
standing cost — **reported findings, not an accepted class**, which is the oracle spec's claim:
contract-surface-gap. A 4000-seed campaign put that class at 8 seeds in 28 findings.

**The validity caveat in `goal.md`'s acceptance-frame section bounds every number here**:
these rates are over the generated grammar, which is not query space, and no denominator that
means anything exists yet (claim: coverage-denominator, `[PROPOSED]`). "52.8% of a grammar" is
a real measurement of a synthetic population and converts into nothing about the SQL people
write.

---

## 6. The refusal histogram and refusal quality {#refusal-histogram}

**Top of the histogram**, seeds 0-1999, of 944 refusals: `WITH` 104, `comparison on
BOOLEAN` 86, `bind error: bad integer literal` 43, `shape=map` WHERE-drop 42, `udf udf0: a
width-1 list return` 42, `modifier on scalar call abs` 39, `QUALIFY` 36, `DISTINCT` 30,
`ORDER BY` 23. Only the first is a whole-relation shape near the top; `QUALIFY` ranks
**seventh**, not third. That is a fact about the grammar, not a demand signal — the campaign
says which classes the *generator* reaches, not which users need
(bears on ask: next-query-classes, in the left-open section).

Per-row shares the scope rows and gap entries carry: `WITH` 104 of 944, `QUALIFY` 36,
`DISTINCT` 30, `ORDER BY` 23 (exclusion: whole-relation-shapes); `shape='map': a WHERE
clause can drop ...` 42 of 944 (exclusion: multiplicity-by-default, its permanent half);
`UNSHIPPED` 14 of 2000 (gap: unshipped-decimal-arithmetic);
`DIVERGE_OPT` 7 of 2000 (exclusion: optimizer-on-answers); `comparison on BOOLEAN` 86 of 944
(gap: undocumented-boolean-comparison, which belongs to no row at all).

**claim: refusal-prefix-share.** Refusal *quality* is measurable today and had never been
measured. Of 944 refusals, **837 (88.7%)** carry one of the three documented prefixes
(`unsupported:` / `parse error:` / `bind error:`); **107 (11.3%)** carry none. **688
(72.9%)** fall in the corpus gate's narrower `_CLEAN` set, which excludes `bind error:`
entirely — and admits two families that are not prefixes at all: `_CLEAN` is
`("unsupported:", "parse error:", "duplicate map key", "NULL in value column")`
(`test_corpus_replay.py:36`), whose last two are *substrings* and appear in neither
`known-limitations.md:279-282`'s three-prefix list nor anywhere else in the docs. The corpus
gate's own definition of clean is already wider than the documentation.

**What 88.7% measures, and what it does not.** It measures *prefix presence*, not that the
message names a construct — those are different properties, and the weaker one is the one
with a number. A correlated subquery is the demonstration: it refuses, correctly, but
through the frontend's catch-all `other => Err(unsup(format!("expression: {other}")))`
(`frontend.rs:3551`), which echoes the AST text rather than naming the construct, and there
is no subquery-specific refusal site to find. It counts inside the 837 and inside the 688
all the same. `goal.md`'s kpi: named-refusal-share is defined on the *stronger* property, so
88.7% is a **floor** for it, not a reading of it.

The undocumented 107 are three families, and **two of them are not named in the oracle
spec's existing correction**: `shape='map': a WHERE clause can drop ...` (42) and its
static-only twin (1); `udf '<name>': a width-1 list return ...` (63); `static data mismatch:
@0: duplicate ...` (1). The oracle spec's claim: refusal-message-prefixes already records
that the prefix set is not exhaustive and routes the fix through
ticket: clean-prefix-reconcile; these two families are new members for that ticket. This is
gap: undocumented-refusal-prefixes.

**The refusal-site census.** 281 `unsup(...)` call sites across nine source files, but
**only ~166 of them are the scope ledger's business**. `specializer/frontend.rs` 144 and
`specializer/retrans.rs` 22 raise `PrepareError` — what `DuckDBInferFn` refuses (two of the
166 are the `fn unsup` definitions themselves, `frontend.rs:64` and `retrans.rs:11`). The
other 115 live under `src/dialect/` (`duckdb.rs` 84, `plan.rs` 13, `bigquery.rs` 7,
`spark.rs` 6, `printer.rs` 2, `ty.rs` 2, `mod.rs` 1) and raise `DialectError::Unsupported` —
a different error type on the translation surface feeding the L2/L3 gates
(`src/dialect/mod.rs:19-24`), which claim: dialect-gate-oracle scopes out. So the gap is
~164 call sites against six scope rows plus this ledger's entries, not 281 against ten. **No
mechanism maps either set to a row.** That is an open question of this reading (the left-open
section), not of `goal.md`'s ask: exclusion-ratification, which asks whether the *permanent*
set binds.

---

## 7. The mined corpus {#mined-statements}

**claim: corpus-match-today.** Of the 678 statements mined from DuckDB's own test suite,
**547 replay bit-exact, 131 refuse cleanly, 0 FAIL**. Zero FAILs is the gate and it holds;
the match count is deliberately ungated, and it has slipped three below the last number
written down — gap: corpus-match-slip carries that and the cleanup it implies.

*Read from:* `packages/confit/tests/test_corpus_replay.py:171-190` (the zero-FAIL gate,
`:180-184` prints the match count, `:186` asserts only `not fails`);
`packages/confit/docs/oracle/` claim: zero-fails-gate and its correction.

---

## 8. Dialect floors {#l2-and-l3-floors}

**claim: dialect-floors-today.** The dialect frontend's L2 gate (parse then print is
invisible to the oracle) stands at **288/678 match, 390 clean-unsupported, 0 FAIL** —
exactly on its floor of 288 (`packages/confit/tests/test_dialect_corpus_gate.py:37`).

The L3 cross-engine gate's Spark floor is **260**
(`packages/confit/tests/test_dialect_cross_engine_gate.py:50`) and **could not be measured
here**: `pyspark` is not installed in this environment, and the fixture fails loudly by
design rather than skipping. Ratcheting that floor means naming the environment that checks
it (bears on `goal.md`'s kpi: ladder-ratchet).

Both floors already carry the ratchet — "raise it when the surface grows, never lower it" —
which is why two of the four ladders in `goal.md`'s four-yardsticks table are enforced and
two are not.

---

## 9. The authoring-side ladder {#authoring-side-ladder}

`packages/confit/docs/goal.md` kpi: coverage-ladder pins the sql-transform admission
ladder: **11 marginalized + 11 refused of 22 mined**, and **39 marginalized / 17 refused /
5 schema-mode** curated. Verified fresh 2026-09-02, and it takes **two** tests, not one:
`_corpus_test.py::test_progression_totals` (`:232-239`) pins the totals — `len(MINED) == 22`,
the two mined buckets summing to 22, and the three curated lengths — while the **11/11 split
itself** is pinned only by `::test_mined_corpus_scoreboard` (`:197-207`,
`counts == MINED_SCOREBOARD`). A drift to 12/10 passes the first and fails the second.

That ladder is `sql_transform`'s, not confit's (goal: engine-half-only); it is read here
because acceptance growth is measured on both sides of the boundary, and it is the only one
of the four ladders whose *split* has a pin test that fails when the number moves. Its named
headroom is gap: admission-ladder-headroom.

---

## 10. Serving latency {#latency-reading}

**claim: bench-is-stale.** The recorded serving-latency table is dated **2026-08-04**
(`kpis.md:118`, the file kpi: serving-latency absorbed and deleted — the old readings are in
git history there) and is **389 commits** behind `2ba96e5`. It carries no commit of its own —
`a6fa318` appears in kpis.md only at `:143`, introducing the *transformer*-path tables, and
attaching it to the pure-SQL table would be a misreading. Worse than age: its headline
claim — "~1.5-1.7x faster than a handwritten Python microservice twin" — compares against a
harness row that **no longer exists**. The record of that retirement is in the harness, not
in kpis.md (which has zero occurrences of `python_dict`, `spec_dict` or `pydantic`):
`benchmarks/bench_serving.py:22-25`, "the old typed-model `python` and `spec_dict` rows
retired with the pydantic surface — dict IS the output now".

Re-measured 2026-09-02 (parity gate green on all five scenarios), the comparable pair is
`spec` against the surviving `python_dict` row, p50 ns at n=1:

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
like a tie carries the **second-worst** ratio in the table. **(ii)** The 2026-08-04 pure-SQL
table (`kpis.md:126-131`) has **four** rows; `feature_bundle` has no 2026-08-04 baseline at
all. The comparison is four-against-four plus one new scenario, so "uniform across five"
borrows strength from a row with nothing to be uniform with.

**The regime holds; the Python comparison's sign does not.** `spec` p50 at n=1 is
2,300-8,800 ns per call across the five scenarios against 5.3-13.1 ms for DuckDB-per-call on
the same rows — the three-orders-of-magnitude gap that goal: request-latency-budget calls a
regime is intact, and that goal is `Unverified` by construction: no gate, floor or pin
bounds serving latency, so a regression that moved this number would fail nothing.

This is **not** called a regression here. The baseline row changed identity (plain dicts
out, where the old row returned pydantic models), the machine is not the 2026-08-04 machine,
and kpi: serving-latency warns absolute numbers drift with load. That warning is not
theoretical: `store_sales`' `spec` cell read 5,600 and 8,800 ns in two runs on the same
machine on the same day, a 57% spread — which is exactly why `goal.md`'s
kpi: bench-refresh-cadence proposes recording a *ratio*. Which of four causes produced the
sign flip is gap: bench-baseline-flip.

---

## 11. KPI enforcement, as read {#enforcement-as-read}

**claim: kpi-pointers-resolve.** Every enforcement pointer in
`packages/confit/docs/goal.md`'s controls-in-force section resolves on master `2ba96e5`,
checked 2026-09-02: C1 `_projection_test.py::gate` (`:34`) and its `MARGINALIZE_FUZZ_N`
seeded differential (`:374-376`, seed 20260729); C2 the `test_duckdb_*.py` wave suites,
`test_params_joins.py`, `test_udfs.py::udf_check` (`:240`), `known-limitations.md`, and
`src/specializer/exec/tests.rs`; C3 `_serving_test.py::serve_gate` (`:29`); C4
`_transformers_test.py::_reference` (`:34`); C5 `_corpus_test.py` and its three-outcome
FAILED-must-be-empty rule. A pointer resolving is not a bar holding at its written depth —
finding: c1-depth is the one place they differ today.

### 11.1 Other dated readings {#other-dated-readings}

Readings that were living in `goal.md` prose and belong to a dated file:

- The standing regexp differential fuzzer re-swept to **zero divergences over 40k cases
  across 8 seeds** (gap: parse-divergence-guards' bell; the gate itself is
  `test_duckdb_regexp_fuzz.py:13-20, :35-37`, N=250, fixed seed).
- `test_arrow_boundary.py` is 120 lines holding four tests (`:38`, `:63`, `:78`, `:100`),
  none of which touches the Arrow batch ceiling; its only mention of it is a prose comment at
  `:33-35`. That is the measurement behind exclusion: resource-ceilings' `Unverified` Arrow
  half, and the live piece of `goal.md`'s ask: exclusion-ratification (1).
- `abs(k) OVER ()` was a silently-dropped modifier on master until it was made to refuse;
  measured 2026-09-02 it raises `unsupported: modifier on scalar call abs`.
- `test_integer_widths.py:856-871` parameterizes `uint8`/`uint16`/`uint32`/`uint64` (and
  `float32`) and asserts the refusal names the column and the type;
  `test_known_limitations.py:180-192` pins the uint64 static the same way. That is what
  gap: wide-integer-lanes' partial bell is, read on 2026-09-02.
- `test_known_limitations.py` holds 14 test functions and its whole-relation
  parameterization (`:98-117`) covers nine constructs — aggregates, `GROUP BY`, `ORDER BY`,
  `LIMIT`, `DISTINCT`, `WITH`, `UNION`, `rowid`, `FULL OUTER JOIN` — while `HAVING`,
  `OFFSET`/`FETCH`/`TOP`, `INTERSECT`/`EXCEPT`, subqueries, multiple statements, table
  functions and `QUALIFY` have no twin there (counted 2026-09-02). This is a sixth site for
  the oracle spec's claim: doc-twin-totality and its open ask: doc-twin-overstatement, not a
  new question.
- `fuzz/gen.py:1060` reads `w = rng.choice([1, 2, 3])  # width-1 list must REFUSE`: the
  generator deliberately emits UDF return shapes it knows are rejected, and those are **63 of
  944 refusals**, roughly **3.2 percentage points** of the 47.2% refusal rate. Changing that
  one `rng.choice` moves acceptance by about three points with no engine change — the
  gameability that `goal.md`'s kpi: acceptance-rate names as its cost, with the number
  attached.
- Findings per campaign, today: **1 per 2000** on the bar (seed 1804) beside **7**
  `DIVERGE_OPT`, with `TIMEOUT`/`PANIC` at 0.
- The `udf '<name>': a width-1 list return is a scalar` class (`src/duckdb/mod.rs:606-609`)
  is **63 of 944 refusals** and fits none of the three grounds — the caller declared their
  UDF wrong. It nevertheless sits in the `REFUSED` bucket that feeds the 52.8%. That is the
  measured body of `goal.md`'s ask: exclusion-ratification (2).

---

## 12. What this reading leaves open {#left-open}

Nothing here is acted on; each item belongs somewhere else. The routing:

| item | where it goes |
|---|---|
| finding: seed-1804 | an xfail-strict pin plus a ticket, per the engine-bug process |
| finding: c1-depth | `goal.md`'s ask: kpi-set-change — correct the text or raise the default |
| finding: static-only-tie-order | a measurement, then a refusal — or a written reason it is acceptable |
| gap: undocumented-boolean-comparison | a bookkeeping fix in `known-limitations.md`, then a row in `goal.md`'s scope-edge section or an entry in the gap-ledger section here |
| gap: undocumented-refusal-prefixes | the oracle spec's ticket: clean-prefix-reconcile, which gains two families |
| gap: bench-baseline-flip | one re-run after `--reinstall-package` rules out (d); then `goal.md`'s kpi: bench-refresh-cadence |
| gap: corpus-match-slip | the oracle spec's ask: match-count-ratchet, answered together with `goal.md`'s ask: acceptance-target |
| the other nine gap entries in the gap-ledger section | ask: next-query-classes below, which is what ranks them |

### 12.1 ask: next-query-classes — which classes are next, and in what order?

*Moved here from `goal.md`'s acceptance-frame section with this reading: every candidate it
ranks is a gap entry rather than a scope decision, so it belongs next to the ledger it ranks.
It is still the owner's question.*

The candidates, each with what it would unlock and what it costs. None of these is
chosen; the list is evidence for a choice, not a plan.

| candidate | unlocks | cost / blocker |
|---|---|---|
| decimal arithmetic | closes gap: unshipped-decimal-arithmetic entirely; empties the only `UNSHIPPED` bucket | an exact decimal lane through the expression tree, not just the join lane that already ships |
| HUGEINT + the unsigned family (the i128 lane) | closes half of gap: wide-integer-lanes; the cranelift dependency was verified GO 2026-08-15 | i128 arithmetic and its overflow traps across both backends |
| narrow-lane overflow traps | closes the one place a narrow lane serves an i64 value on the row path where `infer_arrow` refuses | trap threshold per declared width |
| nested / struct-valued outputs | closes the whole-struct half of gap: non-scalar-values | output schema work at the Arrow boundary |
| `USING` / `NATURAL` self-joins under `shape='many'` | closes half of gap: join-composition-limits | small; it is a named rejection, not a model gap |
| lifting the one-join-per-query restriction under `'many'` | multi-join serving, the other half of gap: join-composition-limits | multiplicity composition, the hardest of these |
| native transform families | closes gap: native-transform-families; a ~100x lever on the transformer path | the per-family ulp bound has to be adopted through review first |

The campaign's refusal histogram (the refusal-histogram section) says which of these the
*generator* reaches, not which your users need, and the ranking it produces is not the
ranking intuition gives — that is a fact about the grammar, not a demand signal. It also holds
a large class that belongs to no entry at all: gap: undocumented-boolean-comparison.

*Binds:* goal: growing-accepted-surface, and every gap entry in the gap-ledger section.

### 12.2 Does the enumeration cover what the engine refuses? {#enumeration-coverage}

*The half of `goal.md`'s ask: exclusion-ratification that asked about coverage rather than
about the target. It is a question about today's code, so it is measured here.*

**Measurably not.** The second-largest refusal class in the campaign appears in no scope row
and nowhere in `known-limitations.md` (gap: undocumented-boolean-comparison), and the
modifier class reached the ledger only by being added to exclusion: whole-relation-shapes in
the pass that wrote `goal.md`. By `known-limitations.md:284`'s own rule a refusal in code and
in no document is a bookkeeping bug to file. Behind that: **~164 `PrepareError` refusal sites
against six scope rows and eleven gap entries, with no mechanism mapping a site to either**
(the refusal-histogram section). The choice is to accept that the enumeration is a curated
summary rather than a cover, or to ask for the mapping — and that choice is not part of
ratifying the permanent set, which is why it is asked here rather than there.

**Reading N=2 replaces none of this.** It is a new file next to it, and the interesting
column is the delta.
