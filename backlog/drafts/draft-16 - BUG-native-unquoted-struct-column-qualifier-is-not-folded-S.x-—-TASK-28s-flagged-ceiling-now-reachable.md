---
id: DRAFT-16
title: >-
  BUG native: unquoted struct-column qualifier is not folded (S.x) — TASK-28's
  flagged ceiling, now reachable
status: Draft
assignee: []
created_date: '2026-07-24 00:36'
labels:
  - native
  - parity
  - bug
  - sql-surface
  - containers
dependencies: []
references:
  - src/expr_build.rs
  - 'tests/test_diff_types.py:150'
documentation:
  - doc-9 (Rich type system and UNNEST — status and deferred edges)
  - doc-1 (DataFusion function catalogue — parity oracle)
priority: medium
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
You have a struct column and you reference it with any capitalization other than all-lowercase:

    SELECT S.x AS v FROM __THIS__     -- struct column `s`, field `x`

DataFusion folds the unquoted qualifier `S` to `s`, exactly as it folds any other unquoted identifier, and returns 7. Native does not fold it and raises "Unknown column: S".

So the same rule users already learned for ordinary columns (unquoted identifiers fold to lowercase, TASK-28) silently does not apply to the struct-column qualifier position. A user who writes `S.x` — or more realistically has a CamelCase struct column and writes `Coords.lat` — gets a query that works on the DataFusion/transform path and fails on native/infer.

WHY THIS IS NOTABLE: THIS IS A PREDICTED CEILING, NOW REACHABLE
TASK-28 (identifier folding) shipped with an explicit acceptance criterion recording this exact gap as an accepted, UNREACHABLE ceiling:

  TASK-28 AC#5: "Ceiling (ponytail note, expr_build.rs): a real CamelCase table/struct-column
  qualifier is NOT folded — unreachable today (tables are always __THIS__/generated); flagged
  for when qualified tables become reachable."

Ritchie's TASK-29 Phase B work (struct field access, landed 671efb2/8d398bf) is what made the qualifier position reachable. So the ceiling did exactly what it was flagged to do: it stayed harmless until a new feature reached it, and the flag is why we recognized it immediately instead of re-diagnosing from scratch.

That is the process working. Worth noting when reviewing: the TASK-28 ponytail note earned its keep.

CURRENT STATE
Pinned by a strict xfail_on_native at tests/test_diff_types.py::test_uppercase_qualifier_field_access. Codegen matches the oracle; native is the outlier. Filed per the standing native-bug process (xfail-strict + ticket, never fix inline) — Ritchie did the xfail half correctly.

SEVERITY
Fails LOUDLY (raises "Unknown column: S") rather than silently computing a different value, so it is in the DRAFT-12 category rather than the DRAFT-13 category. Medium: real, but self-announcing, and the workaround (lowercase or quote the qualifier) is available once you know. Native is the DEFAULT serving engine, which is what keeps it from being Low.

RELATED: DRAFT-12 (native has no struct(...)/make_array(...) dispatch) and DRAFT-13 (native does not widen mixed-numeric list elements) are the other native container gaps from the same Phase B work. Possibly worth scoping all three as one "native container parity" push rather than three separate fixes — they are all in expr_build.rs / types.rs and all surfaced together.

DRAFT pending AmirHossein's review of scope/priority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 native folds an unquoted struct-column qualifier to lowercase, matching DataFusion (SELECT S.x resolves against struct column s)
- [ ] #2 A quoted qualifier stays case-exact, consistent with the TASK-28 folding rule for ordinary identifiers
- [ ] #3 The xfail_on_native marker on tests/test_diff_types.py::test_uppercase_qualifier_field_access is removed and the test passes on both engines
- [ ] #4 TASK-28 AC#5's ceiling note in expr_build.rs is updated or removed, since the gap it flagged is now closed rather than merely unreachable
<!-- AC:END -->
