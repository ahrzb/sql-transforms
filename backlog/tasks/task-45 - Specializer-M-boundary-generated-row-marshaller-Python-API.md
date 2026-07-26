---
id: TASK-45
title: 'Specializer M-boundary: generated row marshaller + Python API'
status: In Progress
assignee: []
created_date: '2026-07-25 02:32'
updated_date: '2026-07-26 08:00'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stretch plan (recorded 2026-07-26, design doc §3 flag 1 + §10). Measured targets from the M-cranelift bench: per-cell getattr with a fresh name string, per-call buffer allocs, and model_validate per output row — those three ARE the boundary that dominates end-to-end.
1. Marshaller core (src/duckdb, the boundary module): prepare-time interned PyString field names for input attrs and output keys; input rows accepted as dict (get_item on interned key) or pydantic model (getattr on interned name), unboxed type-directed straight into the existing Batch columns — SoA stays: both backends read Batch, the batch is L1-resident at this n, and the doc's own flag-1 argument makes layout irrelevant here (deviation from the AoS row-struct line, noted deliberately); output built model_construct-style (cached bound method + interned keys), never model_validate; RunState + input buffers owned by the fn object behind a Mutex, cleared not dropped per call. SPECIALIZER_GENERIC_BOUNDARY env knob keeps the old generic path runnable for the baseline; a .boundary getter mirrors .backend.
2. SpecializedTransform (sql_transform package): SQLTransform API minus transformer refs — ctor(sql | Template), fit(table, this_model=None) reusing desugar/inline_references/build_state_tables/rewrite_sql then preparing DuckDBInferFn (clear error on transformer refs; records output only, dense raises); infer/infer_batch pass dicts/models straight through, no SimpleNamespace hop. Window aggregates rewrite into static-table equi-joins = v0 subset, so they ride along where the rewrite output parses.
3. Bench per §10: extend scripts/bench_specializer.py — no-op f through generic vs marshalled boundary (AC #2, the marshaller's win as a measured number), end-to-end p50/p99 at n in {1,8,64,1024} vs native + codegen (AC #3); numbers into this ticket.
4. Zero-alloc steady state (AC #4): counting-global-allocator Rust test around the reused-state run path asserting no Rust-side allocation on the second call (arena reset only); marshaller buffer reuse asserted the same way. Gate green (AC #5).
<!-- SECTION:PLAN:END -->

