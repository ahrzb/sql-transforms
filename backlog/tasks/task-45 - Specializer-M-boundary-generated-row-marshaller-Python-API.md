---
id: TASK-45
title: 'Specializer M-boundary: generated row marshaller + Python API'
status: To Do
assignee: []
created_date: '2026-07-25 02:32'
labels: []
milestone: m-7
dependencies:
  - TASK-44
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
type: feature
ordinal: 39000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire the specializer into the Python surface: SpecializedTransform (SQLTransform API minus transformer refs) whose fit() runs prepare and whose infer/infer_batch cross the boundary through a prepare-time-generated marshaller — fixed field order, interned names, packed row structs, model_construct-style output fill (design doc §3 flag 1: input is row-major; no columnar transpose in the hot path). Baseline the boundary with a no-op f through both the generic pydantic path and the generated marshaller so the win is measured, then end-to-end p50/p99 vs the current engines.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SpecializedTransform fit/infer/infer_batch works end-to-end on the v0 subset with dict and pydantic-model rows
- [ ] #2 No-op-f boundary baseline reported: generic pydantic path vs generated marshaller, p50/p99 at n in {1, 8, 64, 1024}
- [ ] #3 End-to-end p50/p99 vs the current native and codegen engines reported
- [ ] #4 Steady-state hot path allocates nothing per call beyond the output objects; arena reset only
- [ ] #5 mise gate-specializer green
<!-- AC:END -->
