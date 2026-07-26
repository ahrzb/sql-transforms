---
id: TASK-45
title: 'Specializer M-boundary: generated row marshaller + Python API'
status: Done
assignee: []
created_date: '2026-07-25 02:32'
updated_date: '2026-07-26 11:40'
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
- [x] #1 SpecializedTransform fit/infer/infer_batch works end-to-end on the v0 subset with dict and pydantic-model rows
- [x] #2 No-op-f boundary baseline reported: generic pydantic path vs generated marshaller, p50/p99 at n in {1, 8, 64, 1024}
- [x] #3 End-to-end p50/p99 vs the current native and codegen engines reported
- [x] #4 Steady-state hot path allocates nothing per call beyond the output objects; arena reset only
- [x] #5 mise gate-specializer green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stretch plan (recorded 2026-07-26, design doc §3 flag 1 + §10). Measured targets from the M-cranelift bench: per-cell getattr with a fresh name string, per-call buffer allocs, and model_validate per output row — those three ARE the boundary that dominates end-to-end.
1. Marshaller core (src/duckdb, the boundary module): prepare-time interned PyString field names for input attrs and output keys; input rows accepted as dict (get_item on interned key) or pydantic model (getattr on interned name), unboxed type-directed straight into the existing Batch columns — SoA stays: both backends read Batch, the batch is L1-resident at this n, and the doc's own flag-1 argument makes layout irrelevant here (deviation from the AoS row-struct line, noted deliberately); output built model_construct-style (cached bound method + interned keys), never model_validate; RunState + input buffers owned by the fn object behind a Mutex, cleared not dropped per call. SPECIALIZER_GENERIC_BOUNDARY env knob keeps the old generic path runnable for the baseline; a .boundary getter mirrors .backend.
2. SpecializedTransform (sql_transform package): SQLTransform API minus transformer refs — ctor(sql | Template), fit(table, this_model=None) reusing desugar/inline_references/build_state_tables/rewrite_sql then preparing DuckDBInferFn (clear error on transformer refs; records output only, dense raises); infer/infer_batch pass dicts/models straight through, no SimpleNamespace hop. Window aggregates rewrite into static-table equi-joins = v0 subset, so they ride along where the rewrite output parses.
3. Bench per §10: extend scripts/bench_specializer.py — no-op f through generic vs marshalled boundary (AC #2, the marshaller's win as a measured number), end-to-end p50/p99 at n in {1,8,64,1024} vs native + codegen (AC #3); numbers into this ticket.
4. Zero-alloc steady state (AC #4): counting-global-allocator Rust test around the reused-state run path asserting no Rust-side allocation on the second call (arena reset only); marshaller buffer reuse asserted the same way. Gate green (AC #5).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All four stretches landed on claude/specializer-m-boundary. The marshaller (src/duckdb/mod.rs) does at prepare time everything knowable at prepare time: interned attribute-name PyStrings in fixed field order, input buffers + RunState owned and cleared-not-dropped per call, dict rows via get_item and model rows via getattr on the interned names, output rows by direct pydantic-v2 slot fill. Two assumptions died by measurement: (1) pydantic's literal model_construct API is pure-Python and SLOWER than model_validate (1432 vs 882 ns/row on 2.13) — the shipped path is object.__new__ + object.__setattr__ of __dict__/__pydantic_fields_set__/__pydantic_extra__/__pydantic_private__ at 491 ns, semantically equal (eq, fields_set, assignment all verified); (2) the design doc's AoS row structs buy nothing at L1-resident n — the marshaller fills the existing SoA Batch directly (deliberate deviation, noted in the plan). AC #4 forced real work: ColData::Str became one flat buffer + spans (killing per-cell Strings AND a hidden per-load clone in the JIT's h_load_str), substr/trim became pure sub-span arithmetic, case mapping streams into the arena (Arena::case_map), number→text formats via Arena::push_fmt with DuckF64 on a stack buffer, and h_probe emits without its per-call Vec + ScalarVal clones — all shared between backends, pinned by counting-allocator tests over a probe/arith fixture and a string-heavy program on BOTH backends (the remaining per-call allocs are the output objects themselves plus pyo3's input-list Vec, i.e. the AC's "beyond the output objects"). SpecializedTransform (sql_transform/_specialized.py) reuses the whole fit pipeline minus transformer refs (ValueError at ctor), so window aggregates ride the equi-join rewrite onto cranelift — parity with SQLTransform asserted row-for-row; WHERE stays rejected at the authoring surface (parse_and_validate), while raw DuckDBInferFn keeps it. Bench (scripts/bench_specializer.py, +generic engine via SPECIALIZER_GENERIC_BOUNDARY): noop p50 marshaller vs generic = 1.1/2.1µs at n=1, 468/1192µs at n=1024 (1.9-2.5x, AC #2); vs shipping engines at n=1024 the specializer is 3.7-4.2x faster than native and 2.4-3.8x than codegen (AC #3); cranelift-vs-interp is now visible end-to-end (arith 339 vs 391µs at n=1024). SPECIALIZER_GENERIC_BOUNDARY + .boundary getter mirror the FORCE_INTERP pattern; infer_rows() is the direct hot entry SpecializedTransform uses.
<!-- SECTION:NOTES:END -->


## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
M-boundary delivered and merged (PR #29, rebase-merged 2026-07-26). Generated row marshaller + SpecializedTransform + allocation-free steady state on both backends, with a 17-agent adversarial fleet's 9 confirmed findings (3 root causes) fixed pre-merge: supplied output models keep model_validate semantics, the generic baseline accepts dict rows, reentrant infer falls back instead of erroring. Post-merge the milestone also grew the realistic serving bench (benchmarks/, PR #29): four famous-problem inference paths (titanic 10->24, ames 43->42, ieee-cis fraud 32->41, rossmann 21->44) under an exact three-way parity gate (specializer == DuckDB == handcrafted twin, pytest-enforced). Measured: the specializer beats the handcrafted-Python typed-model server in 16/16 cells (1.1-2.4x), DuckDB-per-call by 1,200-2,700x at n=1, and the previous native/codegen engines build 0/4 scenarios (IS NULL projections, row-x-dim arithmetic unsupported there). Known remaining gap, deliberately reported: a plain-dict handcrafted server is still 1.3-2x faster — typed-output construction is the next perf lever (parked; SQL support prioritized by AmirHossein 2026-07-26).
<!-- SECTION:FINAL_SUMMARY:END -->
