---
id: TASK-7
title: 'Native float display: [1e-5,1e-4) decimal band vs DataFusion'
status: Done
assignee:
  - Wren
created_date: '2026-07-18 14:28'
updated_date: '2026-07-23 00:53'
labels:
  - rust
  - parity
milestone: m-1
dependencies: []
documentation:
  - doc-1 (DataFusion function catalogue — parity oracle)
priority: low
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual of TASK-1 #1: the rust-parity fix handled 1.0/1e300/CONCAT/|| but MISSED the [1e-5,1e-4) band. Native renders CAST(1e-5 AS VARCHAR)->'1e-5', DataFusion->'0.00001' (also 1.5e-5->'1.5e-5' vs '0.000015'; 9e-5 same). Codegen matches oracle. Site: src/expr.rs float display (decimal vs exponential). Already pinned strict xfail-on-rust on master: tests/test_diff_rust_bugs.py::test_float_display_small_decimal_band (auto-surfaces xpass->fail when fixed).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CAST across [1e-5,1e-4) renders decimal matching DF (1e-5->'0.00001'); the pinned xfail flips to pass
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-19 15:54
---
Dispatched to Wren (2026-07-19) — fresh src/expr.rs context from TASK-32. Kept Low (narrow band, real-world-negligible); dispatched now because it's an annoying xfail residual AmirHossein wants gone. Self-verifying: the fix flips the pinned strict xfail, and removing the marker IS the test.
---

author: Iris (PM)
created: 2026-07-19 16:03
---
Done per Wren's merge 42b9f12. Suite now at 0 xfailed — no pinned native parity residuals left in the differential suite. Oracle-probed before coding (validate-don't-assume), targeted one-line fix, boundary parity added. The annoyance is gone.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Merged to master (42b9f12, fix commit 96ff89b). src/expr.rs display_value now renders |x| in [1e-5, 1e-4) as fixed-point shortest-round-trip (`{}`) and keeps `{:?}` elsewhere. Wren probed the oracle first: DataFusion uses fixed decimal for base-10 exponent E in [-5, 15] and scientific outside; `{:?}` already matched all of that except the single E=-5 band — this residual. One-line branch; codegen untouched (already matched). Test: removed the xfail_on_native guard on test_float_display_small_decimal_band (now a normal passing differential case, 1e-5 -> '0.00001') + added band and just-outside-boundary parity checks (9.99e-6 stays scientific, 1e-4 stays fixed; parity-only, no hand-computed digit strings). Verified: full suite 488 passed / 16 skipped / 0 xfailed — the strict xfail is gone and nothing else flipped to XPASS. Closes the last of the TASK-1 rust float-parity residual family.
<!-- SECTION:FINAL_SUMMARY:END -->
