---
id: TASK-50
title: >-
  Specializer SQL support: join forms — USING, residual ON, semi joins, comma
  rewrite (wave 4)
status: In Progress
assignee: []
created_date: '2026-07-26 18:15'
updated_date: '2026-07-26 19:24'
labels: []
milestone: m-7
dependencies:
  - TASK-49
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
type: feature
ordinal: 44000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Census-itemized (2026-07-26, post-wave-3: 71 join-blocked cases). STAGE A — no new execution model (~43 cases): JOIN USING desugar incl. chains and repeated statics (8), all-key/semi joins by lifting the no-value-columns restriction (4), star expansion over joined tables incl. key columns reconstructed from the dynamic side (2), comma-join rewrite to INNER equi-probe for dynamic-x-static with equi WHERE conjuncts + residual WHERE (13), equi-ON with residual predicates preserving 0-or-1 multiplicity for INNER and LEFT (12: ON l.id=r.id AND l.val>1 / AND true / AND false / constant equalities), cross join to a PROVABLY 1-row static (4). STAGE B (design gate — present before building): output multiplicity (Emit-continuation in the IR + both backends) for pure non-equi ON and range comma-joins (~20 cases). DESCOPED: dynamic self-joins (~9, needs batch-as-build-side), FULL OUTER (1, build-driven output), NATURAL (2, deeper-blocked by COLUMNS()). Pins fleet FIRST: USING output-column semantics (dedup, qualification), LEFT-join residual-ON vs WHERE placement, ON constant conditions, comma-join equivalence to cross-then-filter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Join-form pins measured BEFORE implementation (USING star/dedup/qualification; residual ON semantics on INNER and LEFT incl. constants; comma-join = cross-then-filter equivalence; 1-row cross join) — JSON + spec addendum committed
- [x] #2 Stage A ships: USING, all-key/semi, star-over-join, comma equi rewrite, equi-ON residuals — both backends, differential green
- [x] #3 Multiplicity (stage B) gets a written design presented at the stop point — NOT built without explicit go
- [x] #4 Corpus replay: flips recorded, zero FAILs
- [x] #5 mise gate-specializer green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Pins committed (7bca38c) BEFORE implementation. Fleet: 3 agents, 223k tokens. Headlines: comma == INNER probe bit-exactly (rewrite safe, WHERE conjuncts split freely); USING merged col stays at LEFT position with LEFT value (exact under Inner/Left), both quals addressable, chained USING binds merged, case-insensitive even quoted, USING-after-ON must ERROR; LEFT keeps residual-failing key-matches with all-NULL side, AND false => all-NULL side; ERROR-EAGERNESS trap: single-side residuals are scan-pushed (eager traps in duck) while both-sides are pair-lazy => v0 requires single-side residuals conservatively trap-free, both-sides may trap; cross-to-1-row = empty-keys probe with compile-time 1-entry check (0-row annihilation is free); duplicate star output names survive to Arrow => our typed model cannot represent them, SELECT * both-keys shapes stay honestly unsupported (revised flip estimate ~30-37, not 43). STAGE-A DESIGN FROZEN: JoinSpec += residual SExpr (hit-guarded lowering, match = hit AND residual), SKind::JoinHit for key-column reconstruction, empty val_cols allowed, empty keys = cross with materialization check, USING desugar in bind_from with using-col star suppression. NO IR changes.

STAGE A COMPLETE. Corpus 383 -> 395 of 678, zero FAILs (deterministic); gate green (753 + 13 xfail); 15 new join oracle tests all green first run. Join-blocked pool 71 -> 13: only dynamic self-joins (10), NATURAL (2, deeper-blocked by COLUMNS()), FULL OUTER (1) remain — the other 58 either match or reclassified to their TRUE deeper blocker (aggregation, NULL-in-value-column for the nulltable cross cases, non-equi -> the cross path where multi-row statics reject via duplicate-map-key, which is CORRECT: a 1-row static would genuinely serve). Implementation: bind_on equality/residual split with the scan_residual totality/sidedness walk; hit-guarded residual lowering threads probe value lanes through block params + the live stack (strict SSA violation caught by corpus replay, fixed); USING desugar with merged-column suppression; key_lane reconstruction (INNER = key expr, LEFT = CASE JoinHit); all-key semi joins via one-side-empty map statics (grammar + verifier extended, round-trip fuzz green); comma rewrite consuming WHERE equi conjuncts with leftover-WHERE reconstruction; UBIGINT/HUGEINT static payloads reject cleanly. STAGE B WRITTEN DESIGN (not built, per AC #3): (1) Term::EmitTo{then,args} — emit current out-row, continue at block (plain Emit stays as row-end); (2) interp: push row + jump; cranelift: emit becomes an h_emit(cx) helper call (per-row ABI return then means only row-done); (3) lowering wraps the row program in a loop over static entries (block-param induction + a ProbeAt inst reading entry idx), residual evaluated per entry; INNER emits per match, LEFT tracks any-match and emits the NULL-side row after the loop; (4) contract change: output multiplicity becomes data-dependent — batch-level API only, breaks SpecializedTransform 1:1 row mapping; (5) cost O(|static|) per row. Unlocks ~13 non-equi/self-adjacent cases + N-row cross.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wave-4 stage A complete: JOIN USING (merged-column semantics measured and mirrored — left position, left value, both quals addressable, chained binding, silent dedup), residual ON predicates on INNER and LEFT (match = key_hit AND residual, hit-guarded per-pair evaluation exactly matching DuckDB's measured laziness; single-side residuals restricted to trap-free shapes because DuckDB scan-pushes those), all-key/semi joins (one-side-empty map statics with text-format round-trip), star expansion over joined tables with key-column reconstruction (INNER = key expr, LEFT = CASE JoinHit THEN key ELSE NULL), comma-join rewrite to INNER probes with WHERE-conjunct key consumption, and cross join to 1-row statics via empty-key maps (0-row annihilates; multi-row rejects via the duplicate-key check). Corpus 383 -> 395 of 678, zero FAILs; join-blocked pool 71 -> 13 (self-joins, NATURAL, FULL OUTER — all named). Stage B (output multiplicity via Term::EmitTo + per-entry static scan) delivered as a written design, not built, per the design gate.
<!-- SECTION:FINAL_SUMMARY:END -->
