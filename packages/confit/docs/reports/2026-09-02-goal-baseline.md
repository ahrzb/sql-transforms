# Goal yardsticks, baseline reading (2026-09-02)

**What this is.** The first reading of the goal document's yardsticks — reading **N=1**.
`packages/confit/docs/goal.md` defines what confit is for, what it excludes, and *how each
of those is measured*; it deliberately holds no number. This file holds the numbers that
document's definitions produce, on one dated run, in one named environment. It is a
**reading, not a rule**: nothing here amends a goal, ratifies an exclusion or adopts a KPI.

**A later reading is a new dated file in this directory**, never an edit to this one. That
is the whole point of the split: a document that carries its own readings in prose is how a
bar goes a month unre-run and nobody notices — which is exactly what §7 below measured.

**Slugs.** Items keep the slugs they had when they lived in `goal.md`; the family is
carried by the citation, so what was written there as an ASK is written here as
`finding:` where the thing is a finding with options rather than a question the owner must
answer. `goal:`, `exclusion:`, `kpi:` and `ask:` citations resolve in `goal.md`;
`claim:` and `divergence:` citations without a local definition resolve in
`packages/confit/docs/oracle/`.

---

## 1. Environment and reproduction

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
engine rows rather than failing. See finding: bench-baseline-flip (d): the harness's own
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

## 2. Campaign verdicts and the acceptance reading

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
`UNSHIPPED` (they build and serve, only the comparison is withheld), the 21 `AGREE_TRAP`,
and the one live `DIVERGE_VALUE`: on this definition the seed-1804 parity defect below
counts as accepted, which is correct for a *scope* metric and is the reason acceptance can
never stand in for parity. All 14 `UNSHIPPED` are class `decimals`
(exclusion: unshipped-decimal-arithmetic). All 7 `DIVERGE_OPT` are
exclusion: optimizer-on-answers' standing cost — **reported findings, not an accepted
class**, which is the oracle spec's claim: contract-surface-gap. A 4000-seed campaign put
that class at 8 seeds in 28 findings.

**The validity caveat in `goal.md` §2 bounds every number in this section**: these rates are
over the generated grammar, which is not query space, and no denominator that means
anything exists yet (claim: coverage-denominator, `[PROPOSED]`). "52.8% of a grammar" is a
real measurement of a synthetic population and converts into nothing about the SQL people
write.

### 2.1 finding: seed-1804 — a live parity defect

The one `DIVERGE_VALUE`, seed 1804: `CAST(pow(-0.25e0, 0.1e0) AS VARCHAR)` inside
`struct_pack` gives `'nan'` on one side and `'-nan'` on the other — a NaN *sign* reaching a
string, which the canonical `repr` comparison elsewhere cannot see (claim: blind-spots' NaN
row). Recorded, not fixed: per the engine-bug process it wants an xfail-strict pin and a
ticket.

---

## 3. The refusal histogram and refusal quality

**Top of the histogram**, seeds 0-1999, of 944 refusals: `WITH` 104, `comparison on
BOOLEAN` 86, `bind error: bad integer literal` 43, `shape=map` WHERE-drop 42, `udf udf0: a
width-1 list return` 42, `modifier on scalar call abs` 39, `QUALIFY` 36, `DISTINCT` 30,
`ORDER BY` 23. Only the first is a whole-relation shape near the top; `QUALIFY` ranks
**seventh**, not third. That is a fact about the grammar, not a demand signal — the campaign
says which classes the *generator* reaches, not which users need
(bears on goal.md's ask: next-query-classes).

Per-row shares the exclusion ledger's rows were carrying: `WITH` 104 of 944, `QUALIFY` 36,
`DISTINCT` 30, `ORDER BY` 23 (exclusion: whole-relation-shapes); `shape='map': a WHERE
clause can drop ...` 42 of 944 (exclusion: multiplicity-by-default); `UNSHIPPED` 14 of 2000,
all class `decimals` (exclusion: unshipped-decimal-arithmetic); `DIVERGE_OPT` 7 of 2000
(exclusion: optimizer-on-answers).

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
all the same. goal.md's kpi: named-refusal-share is defined on the *stronger* property, so
88.7% is a **floor** for it, not a reading of it.

The undocumented 107 are three families, and **two of them are not named in the oracle
spec's existing correction**: `shape='map': a WHERE clause can drop ...` (42) and its
static-only twin (1); `udf '<name>': a width-1 list return ...` (63); `static data mismatch:
@0: duplicate ...` (1). The oracle spec's claim: refusal-message-prefixes already records
that the prefix set is not exhaustive and routes the fix through
ticket: clean-prefix-reconcile; these two families are new members for that ticket.

**The refusal-site census, behind goal.md's ask: exclusion-ratification (2).** 281
`unsup(...)` call sites across nine source files, but **only ~166 of them are the exclusion
ledger's business**. `specializer/frontend.rs` 144 and `specializer/retrans.rs` 22 raise
`PrepareError` — what `DuckDBInferFn` refuses (two of the 166 are the `fn unsup` definitions
themselves, `frontend.rs:64` and `retrans.rs:11`). The other 115 live under `src/dialect/`
(`duckdb.rs` 84, `plan.rs` 13, `bigquery.rs` 7, `spark.rs` 6, `printer.rs` 2, `ty.rs` 2,
`mod.rs` 1) and raise `DialectError::Unsupported` — a different error type on the
translation surface feeding the L2/L3 gates (`src/dialect/mod.rs:19-24`), which
claim: dialect-gate-oracle scopes out. So the gap is ~164 call sites against ten ledger
rows, not 281 against ten. No mechanism maps either set to a ledger row.

**finding: undocumented-boolean-comparison.** `unsupported: comparison on BOOLEAN`
(`specializer/frontend.rs:5322`) is the **second-largest refusal class in the campaign** —
86 of 944, 9.1%, behind only `WITH` — and it appears in **no** exclusion row and **nowhere**
in `known-limitations.md` (`grep -c BOOLEAN` there returns 0). By
`known-limitations.md:284`'s own rule a refusal in code and in no document is a bookkeeping
bug to file. `unsupported: modifier on scalar call abs` (`frontend.rs:6296`, 39 refusals)
reached the ledger only by being added to exclusion: whole-relation-shapes in the pass that
wrote goal.md.

---

## 4. The mined corpus

**claim: corpus-match-today.** Of the 678 statements mined from DuckDB's own test suite,
**547 replay bit-exact, 131 refuse cleanly, 0 FAIL**. Zero FAILs is the gate and it holds;
the match count is deliberately ungated.

**The 550 slippage.** **550** is quoted at six unhedged sites across *five* documents, and
the oracle spec's ladder records 550 as a genuine earlier reading
(`53 -> 395 -> 505 -> 511 -> 529 -> 546 -> 550`), so the ungated match count has moved
**down by three** since it was last written down and nothing noticed. This is fresh evidence
for the oracle spec's open ask: match-count-ratchet — a ladder nothing ratchets has now
demonstrably slipped — and for goal.md's ask: acceptance-target, which should be answered
with it rather than separately.

One note on the cleanup's size, because a ratchet decision turns partly on it: a bare
`grep -rn 550 --include=*.md` on 2026-09-02 also finds the number in
`oracle/09-version-bumps-and-mutability.md:56`, a second occurrence in
`reports/confit-architecture.md:30`, and
`docs/specs/2026-08-13-dialect-logical-plan-design.md:265` and `:298` — so the six
*unhedged current-state* sites the oracle spec's correction enumerates are a subset of the
occurrences a remediation would have to walk.

*Read from:* `packages/confit/tests/test_corpus_replay.py:171-190` (the zero-FAIL gate,
`:180-184` prints the match count, `:186` asserts only `not fails`);
`packages/confit/docs/oracle/` claim: zero-fails-gate and its correction.

---

## 5. Dialect floors

**claim: dialect-floors-today.** The dialect frontend's L2 gate (parse then print is
invisible to the oracle) stands at **288/678 match, 390 clean-unsupported, 0 FAIL** —
exactly on its floor of 288 (`packages/confit/tests/test_dialect_corpus_gate.py:37`).

The L3 cross-engine gate's Spark floor is **260**
(`packages/confit/tests/test_dialect_cross_engine_gate.py:50`) and **could not be measured
here**: `pyspark` is not installed in this environment, and the fixture fails loudly by
design rather than skipping. Ratcheting that floor means naming the environment that checks
it (bears on goal.md's kpi: ladder-ratchet).

Both floors already carry the ratchet — "raise it when the surface grows, never lower it" —
which is why two of the four ladders in goal.md §2.1 are enforced and two are not.

---

## 6. The authoring-side ladder

`packages/confit/docs/goal.md` kpi: coverage-ladder pins the sql-transform admission
ladder: **11 marginalized + 11 refused of 22 mined**, and **39 marginalized / 17 refused /
5 schema-mode** curated. Verified fresh 2026-09-02, and it takes **two** tests, not one:
`_corpus_test.py::test_progression_totals` (`:232-239`) pins the totals — `len(MINED) == 22`,
the two mined buckets summing to 22, and the three curated lengths — while the **11/11 split
itself** is pinned only by `::test_mined_corpus_scoreboard` (`:197-207`,
`counts == MINED_SCOREBOARD`). A drift to 12/10 passes the first and fails the second.

That ladder is `sql_transform`'s, not confit's (goal: engine-half-only); it is read here
because acceptance growth is measured on both sides of the boundary, and it is the only one
of the four ladders whose *split* has a pin test that fails when the number moves.

---

## 7. Serving latency

**claim: bench-is-stale.** D2's serving-latency table is dated **2026-08-04**
(`kpis.md:118`) and is **389 commits** behind `2ba96e5`. It carries no commit of its own —
`a6fa318` appears in kpis.md only at `:143`, introducing the *transformer*-path tables, and
attaching it to the pure-SQL table would be a misreading. Worse than age: D2's headline
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
like a tie carries the **second-worst** ratio in the table. **(ii)** D2's pure-SQL table
(`kpis.md:126-131`) has **four** rows; `feature_bundle` has no 2026-08-04 baseline at all.
The comparison is four-against-four plus one new scenario, so "uniform across five" borrows
strength from a row with nothing to be uniform with.

**The regime holds; the Python comparison's sign does not.** `spec` p50 at n=1 is
2,300-8,800 ns per call across the five scenarios against 5.3-13.1 ms for DuckDB-per-call on
the same rows — the three-orders-of-magnitude gap that goal: request-latency-budget calls a
regime is intact, and that goal is `Unverified` by construction: no gate, floor or pin
bounds serving latency, so a regression that moved this number would fail nothing.

This is **not** called a regression here. The baseline row changed identity (plain dicts
out, where the old row returned pydantic models), the machine is not the 2026-08-04 machine,
and kpi: serving-latency warns absolute numbers drift with load. That warning is not theoretical:
`store_sales`' `spec` cell read 5,600 and 8,800 ns in two runs on the same machine on the
same day, a 57% spread — which is exactly why goal.md's kpi: bench-refresh-cadence proposes
recording a *ratio*.

### 7.1 finding: bench-baseline-flip — regression, or a baseline that changed identity?

D2 records the engine as 1.5-1.7x faster than the handwritten Python twin. Today it is
**1.13-2.05x slower** than the twin row that still exists, on all five scenarios — though
only **four** of them have a 2026-08-04 baseline to have flipped from. Four candidate
readings, in the order they are cheapest to rule out:

**(a) Baseline changed identity.** The old `python` row returned pydantic models; the
surviving `python_dict` row returns plain dicts, which is cheaper. The magnitudes fit:
titanic's Python twin read 6,000 ns in D2 (`kpis.md:128`) and 2,200 ns today. If that
accounts for the whole delta, D2's prose is simply obsolete and the fix is an edit.

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
happened in this repo before, and it is the cheapest of the four to rule out. §1's
environment names `uv run maturin develop --release`, which is not the harness's own
recommended command; ruling (d) out costs one re-run after `--reinstall-package`.

No bisect was run. What can be said: (c) alone is unlikely to produce a uniform sign flip,
(d) is untested and cheapest, and the answer changes what goal: request-latency-budget is
worth as a claim. **Settle this before adopting kpi: bench-refresh-cadence** — a cadence on a
metric whose baseline changed identity would re-record the confusion.

---

## 8. KPI enforcement, as read

**claim: kpi-pointers-resolve.** Every enforcement pointer under the five control KPIs in
`packages/confit/docs/goal.md` §5.2 resolves on master `2ba96e5`, checked 2026-09-02: C1
`_projection_test.py::gate` (`:34`) and its `MARGINALIZE_FUZZ_N` seeded differential
(`:374-376`, seed 20260729); C2 the `test_duckdb_*.py` wave suites, `test_params_joins.py`,
`test_udfs.py::udf_check` (`:240`), `known-limitations.md`, and
`src/specializer/exec/tests.rs`; C3 `_serving_test.py::serve_gate` (`:29`); C4
`_transformers_test.py::_reference` (`:34`); C5 `_corpus_test.py` and its three-outcome
FAILED-must-be-empty rule.

### 8.1 finding: c1-depth — the control is gated two orders of magnitude shallower than its text

C1 says "1,500-2,000-case runs at each widening loop" (kpi: training-round-trip), but
`_projection_test.py:375` is `n = int(os.environ.get("MARGINALIZE_FUZZ_N", "25"))` — the deep
run is opt-in and the standing gate is 25. Under `goal.md` §5.1 a control that only holds at
its written depth when someone sets an environment variable is "an unacknowledged
trade-off", and the remedy is a decision, not an edit: **either correct C1's text to 25, or
raise the default and pay the runtime**. Both are changes to the KPI set, so both route
through goal.md's ask: kpi-set-change.

### 8.2 Other dated readings, relocated here

Readings that were living in `goal.md` prose and belong to a dated file:

- The standing regexp differential fuzzer re-swept to **zero divergences over 40k cases
  across 8 seeds** (exclusion: parse-divergence-guards' bell; the gate itself is
  `test_duckdb_regexp_fuzz.py:13-20, :35-37`, N=250, fixed seed).
- `test_arrow_boundary.py` is 120 lines holding four tests (`:38`, `:63`, `:78`, `:100`),
  none of which touches the Arrow batch ceiling; its only mention of it is a prose comment at
  `:33-35`. That is the measurement behind exclusion: resource-ceilings' `Unverified` Arrow
  half.
- `abs(k) OVER ()` was a silently-dropped modifier on master until it was made to refuse;
  measured 2026-09-02 it raises `unsupported: modifier on scalar call abs`.
- `test_integer_widths.py:856-871` parameterizes `uint8`/`uint16`/`uint32`/`uint64` (and
  `float32`) and asserts the refusal names the column and the type;
  `test_known_limitations.py:180-192` pins the uint64 static the same way. That is what
  exclusion: wide-integer-lanes' partial bell is, read on 2026-09-02.
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
  gameability that goal.md's kpi: acceptance-rate names as its cost, with the number
  attached.
- Findings per campaign, today: **1 per 2000** on the bar (seed 1804) beside **7**
  `DIVERGE_OPT`, with `TIMEOUT`/`PANIC` at 0.
- The `udf '<name>': a width-1 list return is a scalar` class (`src/duckdb/mod.rs:606-609`)
  is **63 of 944 refusals** and fits none of the three grounds — the caller declared their
  UDF wrong. It nevertheless sits in the `REFUSED` bucket that feeds the 52.8%. That is the
  measured body of goal.md's ask: exclusion-ratification (3).

---

## 9. What this reading leaves open

Nothing here is acted on; each item belongs somewhere else. The routing:

| item | where it goes |
|---|---|
| finding: seed-1804 | an xfail-strict pin plus a ticket, per the engine-bug process |
| finding: undocumented-boolean-comparison | a bookkeeping fix in `known-limitations.md`, and goal.md's ask: exclusion-ratification (2) |
| finding: c1-depth | goal.md's ask: kpi-set-change — correct the text or raise the default |
| finding: bench-baseline-flip | one re-run after `--reinstall-package` rules out (d); then goal.md's kpi: bench-refresh-cadence |
| the 550 slippage | the oracle spec's ask: match-count-ratchet, answered together with goal.md's ask: acceptance-target |
| the two new undocumented prefix families | the oracle spec's ticket: clean-prefix-reconcile |
| a static-only `ORDER BY` with ties freezing a nondeterministic order | goal.md's exclusion: whole-relation-shapes names the hole; it wants a measurement and then a refusal, or a written reason it is acceptable |

**Reading N=2 replaces none of this.** It is a new file next to it, and the interesting
column is the delta.
