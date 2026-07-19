---
id: TASK-1
title: Rust InferFn parity regressions vs DataFusion oracle
status: Done
assignee:
  - Developer
created_date: '2026-07-18 13:44'
updated_date: '2026-07-19 01:15'
labels:
  - rust
  - parity
milestone: m-1
dependencies: []
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
10 divergences where infer (Rust) disagrees with transform (DataFusion) on the same input. DataFusion is the oracle. Pin each with a strict xfail-on-rust on MASTER first, then flip. All 10 fixed & merged (rust-parity-bugs -> b1a10bf; suite 211, 0 xfailed); residual float [1e-5,1e-4) band is TASK-7. Full detail + source sites: git history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CAST(float AS VARCHAR) renders like DF (1.0 not 1); also CONCAT and 1e300 -- src/expr.rs:114
- [x] #2 ROUND(int) returns float 3.0 not int 3 -- src/expr.rs:550 + src/types.rs:268
- [x] #3 NULLIF(1, 1.0) returns NULL via numeric coercion -- src/expr.rs:575
- [x] #4 Unary minus supported (SELECT -a / -1) -- src/expr_build.rs:39-42
- [x] #5 String concat || supported -- src/expr_build.rs:205
- [x] #6 COALESCE(int,float) types as float supertype (same root as NULLIF)
- [x] #7 SUBSTR start <= 0 uses Postgres windowing (PRIORITY: fix first)
- [x] #8 NaN = NaN returns True instead of raising
- [x] #9 CAST(str AS BOOL) accepts t/1/yes like DF
- [x] #10 CAST(str AS INT/FLOAT) with surrounding whitespace errors like DF
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All 10 fixed & merged (rust-parity-bugs -> b1a10bf; suite 211, 0 xfailed). Each pinned first as strict xfail-on-rust in tests/test_diff_rust_bugs.py against LIVE DataFusion, then flipped. Residual sub-case on #1 discovered during codegen merge (float display [1e-5,1e-4) band) -> TASK-7. Minor: SUBSTR negative length (e.g. SUBSTR('hi',2,-1)) is unspecified; DF unprobed, impl returns '' -- no divergence surfaced.
<!-- SECTION:NOTES:END -->
