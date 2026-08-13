---
id: TASK-100
title: >-
  Cranelift i128 spike
status: To Do
assignee: []
created_date: '2026-08-14 03:30'
labels:
  - m-8
dependencies: []
type: spike
ordinal: 92000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The spec's hard dependency for m-8 phase 4: verify cranelift i128
legalization on x64 (an i128 add/mul/cmp through the existing JIT
harness) BEFORE the phase is scheduled, so it cannot stall on a surprise.
Approved 2026-08-13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 a minimal i128 kernel compiles and runs through the cranelift backend on x64
- [ ] #2 findings written into the m-8 milestone (go / no-go / workaround)
<!-- AC:END -->
