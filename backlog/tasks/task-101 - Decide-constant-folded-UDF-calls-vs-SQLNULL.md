---
id: TASK-101
title: >-
  Decide: constant-folded UDF calls vs SQLNULL
status: To Do
assignee: []
created_date: '2026-08-14 03:30'
labels:
  - m-8
dependencies: []
type: decision
ordinal: 93000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Campaign residual (seed 601418, the LAST width finding): DuckDB
constant-folds a python UDF call with all-constant args at plan time;
a NULL result is SQLNULL, so (-1) % (udf(...)).f1 types INTEGER there.
We deliberately never execute UDFs at build (user code, side effects),
so ours types from the declared BIGINT field — int64. Values agree.
Options: (a) known-limitations row (never fold user code at build — the
principled stance); (b) fold PURE-declared UDFs only (new API surface —
RED LINE approval needed); (c) gen-side: stop generating all-constant
UDF args (masks a real difference). AmirHossein decides.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 decision recorded; the corresponding row/tag/tests land with it
<!-- AC:END -->
