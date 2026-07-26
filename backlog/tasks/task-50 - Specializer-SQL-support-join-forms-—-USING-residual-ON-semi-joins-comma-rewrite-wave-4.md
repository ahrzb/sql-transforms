---
id: TASK-50
title: >-
  Specializer SQL support: join forms — USING, residual ON, semi joins, comma
  rewrite (wave 4)
status: In Progress
assignee: []
created_date: '2026-07-26 18:15'
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
- [ ] #1 Join-form pins measured BEFORE implementation (USING star/dedup/qualification; residual ON semantics on INNER and LEFT incl. constants; comma-join = cross-then-filter equivalence; 1-row cross join) — JSON + spec addendum committed
- [ ] #2 Stage A ships: USING, all-key/semi, star-over-join, comma equi rewrite, equi-ON residuals — both backends, differential green
- [ ] #3 Multiplicity (stage B) gets a written design presented at the stop point — NOT built without explicit go
- [ ] #4 Corpus replay: flips recorded, zero FAILs
- [ ] #5 mise gate-specializer green
<!-- AC:END -->
