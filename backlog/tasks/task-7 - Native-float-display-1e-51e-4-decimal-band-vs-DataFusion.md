---
id: TASK-7
title: 'Native float display: [1e-5,1e-4) decimal band vs DataFusion'
status: To Do
assignee:
  - Developer
created_date: '2026-07-18 14:28'
updated_date: '2026-07-18 15:20'
labels:
  - rust
  - parity
milestone: m-1
dependencies: []
references:
  - docs/BACKLOG.md
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual of TASK-1 #1: the rust-parity fix handled 1.0/1e300/CONCAT/|| but MISSED the [1e-5,1e-4) band. Native renders CAST(1e-5 AS VARCHAR)->'1e-5', DataFusion->'0.00001' (also 1.5e-5->'1.5e-5' vs '0.000015'; 9e-5 same). Codegen matches oracle. Site: src/expr.rs float display (decimal vs exponential). Already pinned strict xfail-on-rust on master: tests/test_diff_rust_bugs.py::test_float_display_small_decimal_band (auto-surfaces xpass->fail when fixed).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 CAST across [1e-5,1e-4) renders decimal matching DF (1e-5->'0.00001'); the pinned xfail flips to pass
<!-- AC:END -->
