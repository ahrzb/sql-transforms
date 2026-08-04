---
id: TASK-65
title: >-
  Composition: projections as vocabulary members, with partition composition
status: To Do
assignee: []
created_date: '2026-08-04 19:05'
labels: []
dependencies: []
documentation:
  - docs/superpowers/specs/2026-08-04-composition-members-design.md
  - >-
    backlog/drafts/draft-24 - Named-outputs-struct-to-struct-closure-and-the-three-naming-decisions.md
type: feature
ordinal: 58000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
DRAFT-24 loop 5, as ruled 2026-08-04 (partition composition INCLUDED). A
`SQLProjection` becomes a transformer-vocabulary member: called windowed +
bundled + field-addressed; call-site partition keys prepend to every
internal window and params join; fitting the caller fits the member's
internals through the name adapter into caller-owned state (refit-through;
the member object is never mutated; FITTED members refuse — frozen is
DRAFT-25's θ machinery). Lowering is symbolic: the member's authored SQL
re-marginalizes under a namespaced planner (`__cf_m{k}_*` families —
α-renaming by birth, not by rewrite), its plan splices into the caller's
DAG, and field reads β-reduce the member's item expressions into the call
sites. Full design in the spec above.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Member calls are field-addressed transformer spellings; bare member calls and unknown member fields refuse at CONSTRUCTION (T is authored — the select aliases; the P7 fit-time carve-out applies only to the member's internal transformers)
- [ ] #2 Partition composition: call-site keys prepend to every member window and params join; one member fit per (call-site keys × internal keys) group, gated against an independent per-group reference (C4 shape, re-derived through the adapter by hand)
- [ ] #3 Refit-through only: the member object is bit-identical before/after the caller's fit (pinned); a fitted member refuses naming DRAFT-25
- [ ] #4 Namespace-born α-renaming: caller + member each carrying joins, windows, and a transformer produce zero name collisions; the __cf_ input gate and reserved-passthrough invariants hold
- [ ] #5 Recursion and the nesting cap refuse by name at construction
- [ ] #6 Gates: serve_gate on composed projections (C3), corpus/D1 pins untouched, full pytest from root + cargo green
- [ ] #7 Bench: a composition scenario vs its hand-inlined equivalent in bench_transforms; delta recorded in kpis.md D2 (expected ≈ 0 — symbolic inlining, no runtime boundary)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Branch points mapped 2026-08-04: the member kind-check goes BEFORE the
fit/transform duck-check in `_Planner.transformer_ref` (a SQLProjection
duck-passes it today and dies at fit with a raw DuckDB binder error —
measured); the symmetric no-OVER site is `_Planner.scalar_udf`. The
namespaced re-marginalization needs the member's authored SQL retained on
the object. Adapter fit-side: a rename projection of the caller's level
table under the member's column names. Serving-side substitution points:
the member doc's `["__cf_t", col]` column refs. Engine: no changes —
everything lands in the marginalizer/projection layer.
<!-- SECTION:NOTES:END -->
