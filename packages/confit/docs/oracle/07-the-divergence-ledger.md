# Divergence ledger

This chapter indexes recorded divergences and the evidence behind them. It does not turn
a proposed disposition, a passing test, or a directory name into an owner ruling.

## Historical executable-ledger practice

**claim: keep-vs-change.** The repository historically separated intent as follows:

- `tests/known_divergences/` was the **KEEP** ledger: passing tests documented behavior
  with a measured reason.
- `tests/test_open_divergences.py` was the **CHANGE** ledger: one ticketed
  `xfail(strict=True)` pin per entry, deleted rather than rewritten when fixed.

This describes the executable-ledger practice; it does **not** approve every indexed
behavior below. **claim: strict-xfail** names the useful mechanism: an unexpected pass
fails rather than silently certifying an unreviewed change. **claim:
empty-change-ledger** records only that the CHANGE file had no pinned entries on
2026-08-25, with successor tickets named; it did not establish that no divergences
existed.

*Evidence:* `tests/known_divergences/README.md:19, :35-46`;
`tests/test_open_divergences.py:9-35`.

**claim: keep-entry-reason.** A historical KEEP entry owes a measured reason, and a
DuckDB claim in that reason remains subject to remeasurement. The string-budget entry
was corrected on 2026-08-16: DuckDB was deterministic, while the limit was confit's
resource judgment. `known-limitations.md:205` still repeats the disproved
spelling-dependence explanation; proposed **ticket: string-budget-ground-fix** tracks
the correction.

**claim: feature-in-flight.** A scheduled m-8 phase or ticket is temporary feature work,
not an accepted permanent divergence. Completion replaces strict xfail with a parity
test, removes the limitation and fuzzer marker, and certifies only after the suppression
is gone. The visible `UNSHIPPED` bucket is not independently gated; see ask:
float-tolerance-list.

*Evidence:* `docs/specs/2026-08-11-duckdb-type-lattice-design.md:110-131` (owner
decision, 2026-08-11), applying to decimal literal typing, decimal cast rounding, and
narrow-lane overflow.

## Divergence index

**claim: ledger-adjudication.** This is the dated index found by the 2026-08-25 sweep.
A proposed disposition is a recommendation. A blank ruling is **unruled**; neither code,
tests, directory placement, nor the proposal itself supplies approval. Only divergence:
decimal-cast-artifact has a recorded owner ruling.

Severity follows [the four-rung ladder](08-the-severity-ladder.md). Trap elision and the
snapshot baseline are evidence/contract entries rather than engine-versus-oracle value
divergences, but remain indexed where readers expect them.

| divergence | kind / severity | proposed disposition | ruling |
|---|---|---|---|
| **divergence: dedup-on-both-sides** | comparison contract / n/a | `PINNED`, permanent | unruled |
| **divergence: approximate-error-text** | comparison scope / n/a | `UNSPECIFIED`, permanent | unruled |
| **divergence: ilike-nul** | source exclusion / n/a | `UNSPECIFIED`, permanent | unruled |
| **divergence: trap-elision** | optimizer-on contract gap / 1 | `PINNED`, permanent | unruled |
| **divergence: nan-sign-per-platform** | platform-dependent answer / n/a | `IMPL-DEFINED`, permanent | unruled |
| **divergence: schema-qualifiers** | name resolution / 3 and 4 | `PINNED`, permanent | unruled |
| **divergence: decimal-literal-typing** | feature in flight / 2 | tied to ask: float-tolerance-list | unruled |
| **divergence: decimal-cast-rounding** | same literal-typing mechanism / 2 | tied to parent | unruled |
| **divergence: bind-time-constant-refusals** | conservative refusal / 4 | `PINNED`, permanent; cost uncounted | unruled |
| **divergence: regex-size-guard** | one-sided safety guard / 4 | `PINNED`, permanent | unruled |
| **divergence: narrow-lane-overflow** | feature in flight / 3 on row path | `PINNED` until m-8 phase 3 | unruled |
| **divergence: decimal-cast-artifact** | attributed historical residuals / 2 | attributed, not a residual set | **ruled** by ask: unshipped-never-compared |
| **divergence: phase-two-width-residuals** | unreconstructed historical count / unknown | classify before use | unruled |
| **divergence: string-builder-budget** | resource refusal / 4 | `PINNED`, permanent | unruled |
| **divergence: arrow-batch-ceiling** | resource refusal / 4 | `PINNED`, permanent | unruled |
| **divergence: snapshot-baseline** | evidence hygiene / n/a | choose evidence lifecycle | unruled |

The sweep covered `known-limitations.md` sections 3 and 5,
`tests/known_divergences/`, and the committed snapshot at master `85b4739`. It is not a
claim that no later entry exists. Slugs remain stable when multiple entries describe one
mechanism.

## Evidence notes

These notes supply the evidence that the concise index intentionally does not repeat.

- **divergence: dedup-on-both-sides.** DuckDB boundary deduplication renames duplicate
  output names on both sides before comparison. Evidence:
  `pins-wave5/dup-names-client-contract.json`; `confit.compare.dedup_names`;
  `test_known_limitations.py:255`.
- **divergence: approximate-error-text.** Corpus comparison checks successful answers
  and error class, not complete error-message identity. Evidence:
  `known-limitations.md:219-224`; `test_corpus_replay.py`; claim: error-texts.
- **divergence: ilike-nul.** `ILIKE` with embedded NUL varies with DuckDB column
  statistics; confit is NUL-transparent, so the source is excluded by name. Evidence:
  `test_corpus_replay.py:40-49`; `pins-wave1/pins_like.json`; claim:
  statistics-dependent-exclusion.
- **divergence: trap-elision.** Optimizer-on DuckDB removes a trapping subexpression;
  optimizer-off DuckDB and confit evaluate it. Confit agrees with the oracle, while the
  user-visible optimizer-on surface is reported as `DIVERGE_OPT`. Evidence:
  `known-limitations.md:231-257`;
  `known_divergences/test_trap_elision.py`; the dated snapshot subject to divergence:
  snapshot-baseline.
- **divergence: nan-sign-per-platform.** `%`-by-zero NaN sign follows platform libm; the
  measured contract is per-platform bit agreement, not a universal sign. Evidence:
  `test_duckdb_wave3_mathtail.py:204-232`; `pins-wave3/math_tail.json`.
- **divergence: schema-qualifiers.** `s1.t1` resolves by bare table name where DuckDB
  refuses (severity 3); `w.w.w` takes a longer schema-like parse and confit refuses where
  DuckDB serves (severity 4). `known-limitations.md:260-272` supports the latter only.
  The 2026-08-25 inventory found no executable twin.
- **divergence: bind-time-constant-refusals.** Confit refuses some trapping constants at
  construction even when `WHERE FALSE` or empty input would prevent evaluation; this is
  not a blanket refusal of `WHERE FALSE`. Two owner-accepted measurements appear in
  `rfcs/2026-08-19-keep-the-bind-time-refusals.md:29-58`, but no twin measures the
  DuckDB-serves cost, and the proposed ledger status remains unruled. See ask:
  refusal-cost-counting.
- **divergence: regex-size-guard.** Confit's guard can fire before DuckDB's RE2 limit; it
  may over-refuse but cannot serve a query DuckDB rejects. Evidence:
  `pins-waveB/fuzzer-20260728.json`; `pins-first-methodology.md:79`.
- **divergence: string-builder-budget.** Confit refuses a literal `pad`/`repeat` capable
  of exceeding 1 GiB. DuckDB was deterministic when measured 2026-08-16 (`repeat`
  through 4294967295; `lpad`/`rpad` binder-error above INT32), so the reason is resource
  policy, not spelling dependence. Evidence:
  `known_divergences/test_string_budget.py:108-132`.
- **divergence: arrow-batch-ceiling.** Matching DuckDB's `pa.string()` 32-bit offsets
  implies a 2 GiB-per-batch ceiling. The cited
  `known_divergences/test_arrow_boundary.py:34-36` is only a comment; enforcement was
  **unverified** on 2026-08-25.
- **divergence: decimal-literal-typing / decimal-cast-rounding.** The row path types
  DECIMAL literals as f64, producing an `UNSHIPPED` schema difference and, for
  `CAST(-2.5 AS BIGINT)`, `-2` instead of DuckDB DECIMAL's `-3`. The casts agree when
  given the same input type. The earlier 1-ulp value difference came from a deleted
  harness cast. Evidence: `known-limitations.md:166-174`; `fuzz.oracle._type_delta`.
- **divergence: narrow-lane-overflow.** The row path serves an i64 value where the
  intended narrow lane should overflow; `infer_arrow` refuses it by name. Evidence:
  `known-limitations.md:177-188`; `test_integer_widths.py`. An accepted until-fixed
  disposition would require a strict-xfail twin.
- **divergence: decimal-cast-artifact.** In the 2026-08-17 baseline, seeds 869, 1554,
  and 3269 were 1-ulp deltas created by the harness cast and now classify `UNSHIPPED`;
  seed 998 was the signed-zero face of literal typing; seed 2668 was closed TASK-122.
  The snapshot contained 28 findings when counted 2026-08-25: 16 `DIVERGE_BUILD`, 7
  `DIVERGE_OPT`, 4 `DIVERGE_VALUE`, and 1 `DIVERGE_TRAP`. The ruling keeps normalization
  outside the oracle answer and verdict. Evidence: `findings.jsonl`;
  `2026-08-17-fuzz-triage.md:62-63`; claim: unshipped-verdict.
- **divergence: phase-two-width-residuals.** The quoted “79 of 84” was not found in
  `docs/`, `backlog/`, or committed `findings.jsonl` on 2026-08-25. It has no defect or
  severity meaning until remeasured and classified.
- **divergence: snapshot-baseline.** `findings.jsonl` is a dated 2026-08-17 snapshot,
  not live evidence. An 8% static draw changed RNG addressing, while closed TASK-121 and
  TASK-122 seeds remain. After generator changes, stored SQL—not a seed—is the
  reproducible identity. Evidence: TASK-129 `:148-150`; TASK-121 `:84`; TASK-122
  `:86-88`.

TASK-121's ambiguous-reference family is deliberately not another divergence row: its
note says all 78 findings in the 20k campaign reclassified `REFUSED`, but acceptance
criteria remain unchecked and the snapshot retains 16 seeds. Proposed **ticket:
ambiguity-class-closed** corrects the conflicting dated triage prose.

## Executable-twin coverage

**claim: doc-twin-totality.** The executable-twin mechanism is partial, not total.
TASK-95 remains open; the 2026-08-25 inventory found twins across
`test_known_limitations.py`, `test_arrow_schema_api.py`, `test_corpus_replay.py`,
`test_duckdb_wave3_mathtail.py`, and `known_divergences/`, while schema qualifiers had
none. Removing an in-code admission during the `UNSHIPPED` change did not establish
totality.

**ask: doc-twin-overstatement** remains open: either complete TASK-95 and its totality
check or narrow all prose to the enumerated executable coverage.

## Proposed governance and open decisions

**claim: divergence-placement.** **[PROPOSED]** Put a deliberate divergence beside the
requirement it violates and keep this chapter as its index. No ruling adopts that
placement rule; ask: proposed-rules-adoption owns it.

- **ask: unlisted-divergence** — decide whether every unlisted divergence is a bug. This
  would make index completeness contractual without creating an SLA.
- **ask: tentative-bucket** — decide whether measured but unruled facts receive a
  `tentative` tag or must be classified before campaign closure. Current candidates are
  the future-family threads disposition, phase-two width residuals, and the snapshot
  baseline. The retired static-only fold is not a candidate; see
  [the decision record](../decisions/trustworthy-fold.md).
- **ask: baseline-as-evidence** — regenerate `findings.jsonl` on a cadence, freeze it as
  named dated history, or replace it with dated triage reports. In every option, stored
  SQL is the identity after generator changes.

The compact status of every ASK is in [the decision index](12-ask-index.md).
