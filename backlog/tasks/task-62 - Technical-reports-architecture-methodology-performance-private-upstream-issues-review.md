---
id: TASK-62
title: >-
  Technical reports: architecture, methodology, performance (+ private
  upstream-issues review)
status: Done
assignee: []
created_date: '2026-07-28 04:09'
updated_date: '2026-07-28 04:51'
labels: []
dependencies: []
type: docs
ordinal: 55000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write up the specializer work as standalone technical reports under docs/reports/: (1) architecture of the SQL specializer, (2) pins-first methodology (bug specifics redacted pending review), (3) performance report. Two further documents are PRIVATE pending AmirHossein's review — the DuckDB field-notes catalog and an upstream-issues report (DuckDB bugs found, sqlparser-rs gaps + upstream-fix feasibility, issues in other projects) — delivered in-chat only, NOT committed or published.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Three public reports committed under docs/reports/ and PR opened
- [x] #2 No DuckDB/upstream bug specifics in any committed or published text
- [x] #3 Field-notes catalog + upstream-issues report delivered to the user in chat only
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Shipped as PR #49 (merged, master 16a8539).

**Public — `docs/reports/` (3 documents, ~50k chars):**
- `architecture-of-the-sql-specializer.md` — the partial-evaluation engine model end to end: token pre-rewrites → binder → IR → verifier (its six rules quoted) → interpreter/cranelift backends; the null-lane design (no nullable SSA type, so 3VL bugs are unrepresentable rather than merely tested); strict block-param SSA; the shape contract as static proof; stage-B multiplicity (emit.to loops, multimaps, batch-side self-joins); the boundary layers; and the deliberate omissions with the engine-model reason for each.
- `pins-first-methodology.md` — the discipline: two-outcome contract, verbatim-pins fleets (including the wave-3 over-generalization incident that created the rule), the three-outcome corpus ladder (53→550 of 678, zero FAILs at every rung), the executable limitations twin, the standing differential fuzzer, and an honest accounting of what the discipline costs.
- `performance-report.md` — boundary decomposition (~262 ns/row floor, ~37 ns per output column, ~1.7 µs/row compute vs the twin's 2.2 µs total), us-vs-DuckDB per call (2055× at n=1; crossover ~2–3k rows), the arrow boundary, and the columnar core's build-measure-close arc.

Each is also published as a private artifact for phone reading.

**Private — deliberately NOT committed or published (AmirHossein's instruction, 2026-07-28: "Don't publish the bug related stuff, just in case, I need to review"):** a DuckDB field-notes catalog and a bug/upstream-issues report were delivered in chat only, with every repro re-executed live against DuckDB 1.5.5 rather than quoted from the pins.

Two findings from that re-verification, both worth keeping:
1. **A real correction to our records:** the leading-`$` row-path bug is confined to `regexp_matches`. `regexp_replace`, `regexp_extract`, `regexp_full_match` and `~` were measured consistent between the folded and runtime paths, so the bug corrupts booleans and row sets (a `WHERE regexp_matches(s,'$h')` returns every row) but never rewrites string data. Our earlier notes implied a wider blast radius.
2. **A pin wrongly suspected, then vindicated:** the LIKE column-pattern pathology (`pins-wave1/pins_like.json` pin[60]) initially failed to reproduce and was nearly "corrected". Re-running it exactly as recorded — including the `input_desc` field, `s='a'*n+'b'`, which the first attempts had missed — reproduces it precisely: 0.028 s / 0.723 s / 21.7 s at n=64/128/256 against the recorded 26 ms / 0.69 s / 23.1 s. The pin stands unchanged; the lesson is that a pin must be replayed from its whole record, not from its `expr` line.

The public reports were redacted before commit: the methodology report describes the fuzzer catches only at the "a family whose evaluation paths disagree — rejected by name" level already public in `docs/known-limitations.md`, with no repro details. Upstream filing (4 candidate DuckDB issues, ~5 PR-able sqlparser-rs dialect gaps, 1 Backlog.md bug) is gated on AmirHossein's review and was NOT started.
<!-- SECTION:FINAL_SUMMARY:END -->
