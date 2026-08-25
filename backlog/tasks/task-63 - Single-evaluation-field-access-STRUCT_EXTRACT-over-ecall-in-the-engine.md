---
id: TASK-63
title: >-
  Single-evaluation field access: STRUCT_EXTRACT over ecall in the engine
status: Done
assignee: []
created_date: '2026-08-04 12:40'
updated_date: '2026-08-08 03:38'
labels: []
dependencies: []
documentation:
  - >-
    backlog/drafts/draft-24 - Named-outputs-struct-to-struct-closure-and-the-three-naming-decisions.md
  - packages/confit/docs/kpis.md
type: feature
ordinal: 56000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
DRAFT-24 loop 4, the thorough half. Addressing k output fields of one fitted transformer currently costs k evaluations of that transformer: field access is resolved at marginalize time into k independent width-1 lane UDFs (`__cf_tf0_g0`, `__cf_tf0_g1`, ...), each of which re-runs the whole `transform()` and keeps one lane. A bare width-k item has the same shape after loop 3 expands it.

A one-slot memo on `PythonTransform` (last key -> last result) already covers the **row** serving path, where sibling lane calls for one row are consecutive: measured 138.6us -> 81.0us per row for a 2-field PCA, converging on the single-field cost. It is sound because a UDF is deterministic and pure by contract (P15), so a hit returns the identical tuple. That is the cheap 80%.

The memo does **not** cover the DuckDB batch path, which evaluates column-at-a-time: sibling lane calls are separated by a whole column of rows, so every one of them recomputes. It also does nothing for the general case of one extern's outputs feeding another's inputs.

The thorough fix is in the engine: teach the frontend to bind STRUCT_EXTRACT (or a lane-select) over an `ecall`, reading the k SSA values one `ecall` already produces, instead of minting k separate externs. Then the marginalizer emits ONE call plus k field reads, both serving paths evaluate the transformer once per row, and the same mechanism is the prerequisite for composition (DRAFT-24 loop 5): extern -> extern lane wiring is exactly "read lanes off an ecall and feed them as arguments".

Measured context (packages/confit/docs/kpis.md D2, 2026-08-04): this is a ~35% lever on multi-field transformer queries. It is NOT the dominant one — 93% of a fitted transformer's per-row cost is sklearn's own `transform()` (~61us for a single row vs ~1.5us for the same arithmetic in numpy), which is DRAFT-23's native families, a ~100x lever. Sequence accordingly: this task removes duplicate calls; DRAFT-23 removes what each call costs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The frontend binds STRUCT_EXTRACT (or lane-select) over an `ecall`, reading the k SSA values of one call; the "width-k must be a bare SELECT item" refusal relaxes to allow lane reads while still refusing a width-k value used as a scalar
- [x] #2 The marginalizer emits one call plus k field reads instead of k lane UDFs, for both addressed fields and loop-3's expanded bare items
- [x] #3 Both engines evaluate the transformer once per row for k addressed fields, on the row path AND the DuckDB batch path; a counting test asserts the call count, not just the timing
- [x] #4 Every control KPI holds unchanged: round-trip bit-exact, Confit == DuckDB, infer == transform, transformer columns == the sklearn reference, no new silent-success path
- [x] #5 `benchmarks/bench_transforms.py` re-run and packages/confit/docs/kpis.md D2 updated with the before/after for tf_fields2 and tf_bare2, including the batch path
- [x] #6 P16 (or a successor note) records that field access reads lanes off one call, and the one-slot memo is removed or kept with its reason stated
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prior art in-repo: the `ecall` instruction already produces k dsts (a whole-call validity lane plus one (validity, payload) pair per declared return) — see `packages/confit/src/specializer/ir/mod.rs` and the shared `call_extern` in `exec/interp.rs` that both backends route through. Nothing new is needed in execution; the work is in binding (`frontend.rs`) plus the sql-transform side that currently mints `__cf_tf{j}_g{m}`.

Watch the boundary interaction with loop 3: `WideOut`/the flat alias-prefixed expansion happens at fit and rewrites the serving AST — the k lane calls it emits are exactly what should become k lane reads of one call.

**Shipped 2026-08-04.** Spec: `packages/confit/docs/specs/2026-08-04-single-eval-field-access-design.md`. The DuckDB side needed no query restructuring: measured, DuckDB CSEs textually identical pure scalar-UDF calls (4 mentions = 1 call/row), so a struct-returning registration + repeated mentions is single-evaluation; confit mirrors this with a frontend site cache keyed on the syntactic call. Counted k→1 in `_single_eval_test.py` (both paths + bare item, interleaved groups value-checked) and `test_udfs.py::test_field_access_shares_one_call_per_row` (engine+DuckDB, 3 rows × 2 paths = 6 calls). D2 ratios: 1.89x→0.97x row, 1.84x→0.92x batch. The parallel session's one-slot memo (row-path-only, lost uncommitted) is superseded, not reintroduced — see DRAFT-24 loop 4.

Status flipped to Done 2026-08-08: it shipped on the 4th with every AC ticked
and only the flag left behind. Re-checked before closing —
`_single_eval_test.py` plus `tests/test_udfs.py`: **20 passed**.
<!-- SECTION:NOTES:END -->
