---
id: TASK-57
title: >-
  Input-marshalling perf lever: close the spec_dict vs python_dict gap on the
  ingest side
status: Done
assignee: []
created_date: '2026-07-28 01:10'
updated_date: '2026-07-28 01:15'
labels:
  - specializer
  - performance
dependencies: []
type: enhancement
ordinal: 51000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bench edition 4/5 showed spec_dict trailing python_dict 1.1-1.4x on 3 serving scenarios — the remaining gap is INPUT-side (per-row, per-column getattr through the Python attribute protocol while the handcrafted twin reads fields directly).

Bench-first: reproduce the gap per scenario (spec_dict vs python_dict, p50 at n in {1, 8, 64, 1024}), then apply the measured lever, then re-bench with the three-way parity gate green.

Candidate lever (validate, don't assume): for rows that are EXACT instances of the registered row-model class, fetch __dict__ once per row and use PyDict lookups for every field instead of k getattr calls — sound because the registered model's fields are plain (no descriptors shadow the instance dict); any other row object keeps today's getattr path, and dict rows already take the dict path. Stacked on task-56-wave-c (the marshaller gained path-walking ingest there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Baseline reproduced: per-scenario spec_dict vs python_dict p50 numbers recorded before any change
- [ ] #2 Ingest fast path lands with correctness preserved: exact-model-instance rows only; dict rows and foreign objects keep existing semantics (tests)
- [ ] #3 Re-bench shows the gap narrowed or closed on the lagging scenarios with no regression elsewhere (spec and interp rows within noise)
- [ ] #4 Three-way parity gate (specializer == DuckDB == handcrafted) green; full pytest + cargo gates green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
MEASURED OUTCOME (2026-07-28 overnight): the candidate lever is DEAD and the ticket closes as a measurement result, deliberately shipping no code.

Baseline (spec_dict vs python_dict, p50 ns/call, release build, n=1/n=64): titanic 1.50/1.52x, store_sales 1.42/1.33x, fraud_txn 1.15/1.20x, house_prices 1.09/1.07x — reproduces the edition-4 gap.

Experiment 1 — __dict__ fast path (exact-model rows: one __dict__ fetch + k dict lookups instead of k getattr): built, benched, NEUTRAL (titanic 1.52/1.47, store 1.44/1.37 — noise). pydantic v2 getattr is already dict-speed. REVERTED per ponytail: no measurable win, no code.

Experiment 2 — component decomposition on titanic (10 in-cols, 31 out-cols, n=64): trivial 1-col query 262 ns/row (the WHOLE boundary floor incl. 10-col ingest); star passthrough (10 out) 594 ns/row -> emit ~37 ns/out-col; full SQL 3127 ns/row; model-rows vs dict-rows 3127 vs 3041 (ingest mode irrelevant). Twin does EVERYTHING in 2188 ns/row. Decomposition: boundary ~260 + emit ~1150 (31 cols) -> COMPUTE ~1.7us/row dominates, plus per-value re-boxing on emit that the twin avoids by pointer-copying passthrough fields.

CONCLUSION: there is no cheap input-side lever — the remaining gap is (a) per-output-value boxing/emit at wide outputs and (b) string-heavy kernel compute vs hand-optimized python dict lookups. The real lever is the standing 'own the columnar path' thesis: a columnar input/output API (pa.Table in, arrays out) that skips per-value Python objects entirely. That is an API-surface decision — PROPOSAL for AmirHossein, not built overnight. Scenario kernel profiling (which ops eat the 1.7us on titanic) is the other follow-up candidate.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed as a measurement result (bench-first worked exactly as intended): reproduced the spec_dict vs python_dict gap (1.07-1.52x across the four scenarios), built and benched the __dict__ ingest fast path — measured NEUTRAL and reverted (pydantic v2 getattr is already dict-speed) — then decomposed the cost: boundary floor ~260ns/row, emit ~37ns/output-col, compute dominates (titanic ~1.7us/row vs the twin's 2.2us TOTAL). No cheap input-side lever exists; the real lever is the columnar-path API (pa.Table in / arrays out, skipping per-value Python objects), which is an API-surface decision parked for AmirHossein. No code shipped; numbers and method in the implementation notes.
<!-- SECTION:FINAL_SUMMARY:END -->
