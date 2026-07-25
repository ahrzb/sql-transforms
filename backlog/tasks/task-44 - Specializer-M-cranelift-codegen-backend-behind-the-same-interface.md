---
id: TASK-44
title: 'Specializer M-cranelift: codegen backend behind the same interface'
status: To Do
assignee: []
created_date: '2026-07-25 02:31'
labels: []
milestone: m-7
dependencies:
  - TASK-43
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
type: feature
ordinal: 38000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cranelift-jit backend for the imperative IR (design doc §7; cranelift-jit 0.126 spike-verified on x86_64-pc-windows-msvc 2026-07-25). StaticRef handles resolve to absolute addresses of prepare-time structures owned by the compiled artifact. Interpreter-vs-cranelift differential on random IR programs plus the full corpus; first ns/call numbers per the measurement discipline (design doc §10).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every v0-subset prepared query compiles under cranelift and agrees with the interpreter backend on the corpus and on randomized inputs
- [ ] #2 Uncovered ops fall back to the interpreter backend rather than failing prepare
- [ ] #3 p50/p99 ns/call reported at n in {1, 8, 64, 1024} against the interpreter control and the existing native + codegen engines
- [ ] #4 mise gate-specializer green
<!-- AC:END -->
