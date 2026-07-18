---
id: TASK-1
title: Rust InferFn parity regressions vs DataFusion oracle
status: To Do
assignee:
  - Developer
created_date: '2026-07-18 13:44'
labels:
  - rust
  - parity
dependencies: []
references:
  - docs/BACKLOG.md
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
10 divergences where infer (Rust) disagrees with transform (DataFusion) on the same input. DataFusion is the oracle. Pin each with a strict xfail-on-rust on MASTER first (the codegen branch has its own test_diff_rust_bugs.py, do not wait on it), then flip. Full detail + source sites in docs/BACKLOG.md 'Rust engine (InferFn) parity bugs'. Realism: #7 first; #8/#9/#10 low.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 CAST(float AS VARCHAR) renders like DF (1.0 not 1); also CONCAT and 1e300 -- src/expr.rs:114
- [ ] #2 ROUND(int) returns float 3.0 not int 3 -- src/expr.rs:550 + src/types.rs:268
- [ ] #3 NULLIF(1, 1.0) returns NULL via numeric coercion -- src/expr.rs:575
- [ ] #4 Unary minus supported (SELECT -a / -1) -- src/expr_build.rs:39-42
- [ ] #5 String concat || supported -- src/expr_build.rs:205
- [ ] #6 COALESCE(int,float) types as float supertype (same root as NULLIF)
- [ ] #7 SUBSTR start <= 0 uses Postgres windowing (PRIORITY: fix first)
- [ ] #8 NaN = NaN returns True instead of raising
- [ ] #9 CAST(str AS BOOL) accepts t/1/yes like DF
- [ ] #10 CAST(str AS INT/FLOAT) with surrounding whitespace errors like DF
<!-- AC:END -->
