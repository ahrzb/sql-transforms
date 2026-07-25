---
id: TASK-42
title: 'Specializer M-interp: closure-compiled IR interpreter (the oracle backend)'
status: To Do
assignee: []
created_date: '2026-07-25 02:31'
labels: []
milestone: m-7
dependencies:
  - TASK-41
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
type: feature
ordinal: 36000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the interpreter backend over the imperative IR (design doc §7): one pre-traversal builds a closure tree, then execution is plain dispatch. This backend is the differential-testing oracle for every future codegen backend and the fallback for uncovered ops — correctness and coverage over speed, never optimized. Depends on M-ir for the IR definitions and verifier.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every IR instruction executes; the M-ir fixture programs all produce expected outputs
- [ ] #2 Runs only verifier-accepted IR; rejects unverified programs
- [ ] #3 No allocation during execution (arena-only for varlen), asserted by a test
- [ ] #4 mise gate-specializer green
<!-- AC:END -->
