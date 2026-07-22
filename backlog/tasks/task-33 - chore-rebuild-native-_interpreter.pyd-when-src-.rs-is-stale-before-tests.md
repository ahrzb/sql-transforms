---
id: TASK-33
title: 'chore: rebuild native _interpreter.pyd when src/*.rs is stale before tests'
status: To Do
assignee: []
created_date: '2026-07-19 15:43'
labels:
  - dev-ex
  - native
dependencies: []
references:
  - src/
  - mise.toml
priority: medium
type: chore
ordinal: 33000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The native _interpreter.pyd can silently run STALE: if any src/*.rs is newer than the built .pyd (after a checkout, rebase, or history rewrite), the suite runs against OLD native code with no signal. This just bit QA — identifier folding (TASK-28) landed 07-19 in src/ but the .pyd was built 07-18, so native ran without folding and produced 14 PHANTOM identifier-test failures until a manual maturin rebuild. Timely: the ongoing master history rewrite makes stale builds likely across sessions. Fix: a pre-test guard (mise pretest hook / pytest conftest / build dep) that rebuilds via maturin when any src/*.rs mtime > _interpreter.pyd mtime, else no-op. Keep it cheap — an mtime compare, NOT an unconditional rebuild on every run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Running the suite auto-rebuilds native when any src/*.rs is newer than the built _interpreter.pyd; an up-to-date build is a no-op (no forced rebuild cost)
- [ ] #2 A deliberately-stale .pyd no longer yields phantom failures — the guard rebuilds first
<!-- AC:END -->
