---
id: TASK-64
title: >-
  Private columns: underscore-prefixed fields never cross the output boundary
status: Done
assignee: []
created_date: '2026-08-04 18:20'
labels: []
dependencies: []
documentation:
  - >-
    backlog/drafts/draft-25 - Fit-transform-split-theta-handles-and-the-oracle-rule-for-transform-semantics.md
type: feature
ordinal: 57000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Named intermediates without output pollution: an output field whose name
starts with `_` is usable by later select items (DuckDB's lateral-alias
semantics — measured 2026-08-04: laterals chain, bind inside windows and
inside `struct_pack`; whole-table aggregates refuse) but never appears in
the output struct.

```sql
SELECT age * 2 AS _d, _d + 1 AS out FROM __THIS__
-- => Struct<out>; serving lowers to SELECT (age * 2) + 1 AS out

SELECT tfm(__THIS__) AS _t, _t.f1 + _t.f2 AS score, name FROM __THIS__
-- name the struct ONCE privately, read fields, ship scalars — lowers to
-- lane reads of one call (single-evaluation machinery, merged)
```

The VALUE of a private column is pure oracle semantics (a lateral alias);
the only owned rule is projection policy at the boundary — same category
as `this_model` column canonicalization. Lowering is β-reduction
(substitution into consumers), which the marginalizer already performs
between chain levels: private columns never exist at serving, so row path
≡ batch path holds trivially and repeated uses dedupe via DuckDB CSE +
confit site sharing. NOTE (draft-25 §open-edges): no NEW dedup beyond
that — further sharing is gated on type-system purity tracking.

Sequenced THIRD (2026-08-04): after struct-valued transform calls and
composition (DRAFT-24 loop 5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DECISION at brainstorm: privacy trigger — output-field-NAME based (uniform: a `_meta` table column via `*` also drops; recommended) vs authored-alias-only. Recorded in the spec with the rejected option's reason
- [x] #2 A projection whose every output field is private refuses by name at construction
- [x] #3 Private columns feed later items, windows, and `struct_pack` bundles per measured DuckDB lateral semantics; inherited refusals unchanged (real column beats alias, forward reference, duplicated-alias reference)
- [x] #4 Serving SQL contains no private column (β-reduction); infer/infer_batch/transform output models exclude them; C3 (row ≡ batch) gated
- [x] #5 The lateral-in-transformer-window refusal is lifted ONLY via substitution into fit-side SQL, or explicitly kept for v1 — decided in the spec
- [ ] #6 Composition: a member's private fields are not addressable by callers (falls out of the output struct; pinned by a test) — parked with TASK-65; the pin test rides on composition landing
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Mechanism exists: `alias_exprs` lateral resolution + `rewrite_items`
β-reduction in `_marginalize.py`; the one concentration of work is
substituting private expressions into transformer-bundle fit SQL (the
current `_check_no_lateral_in_window` refusal site). Measured oracle
behaviors to pin: laterals in windows OK, in plain aggregates ERR, `_x`
identifiers legal.
<!-- SECTION:NOTES:END -->
