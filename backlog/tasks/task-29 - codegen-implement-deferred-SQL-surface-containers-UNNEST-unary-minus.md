---
id: TASK-29
title: 'codegen: implement deferred SQL surface (containers/UNNEST, unary minus, ||)'
status: In Progress
assignee:
  - Ritchie
created_date: '2026-07-18 20:14'
updated_date: '2026-07-23 14:32'
labels:
  - codegen
  - feature
dependencies: []
documentation:
  - doc-9 (Rich type system and UNNEST — status and deferred edges)
  - doc-8 (Composition — deferred slices)
priority: low
ordinal: 29000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
You opted into the codegen engine (native is the default per decision-7, codegen is the opt-in alternative). Your SQL uses container types or a couple of ordinary operators:

    SELECT named_struct('lat', latitude, 'lon', longitude) AS coords FROM __THIS__
    SELECT first_name || ' ' || last_name AS full_name FROM __THIS__
    SELECT -balance AS debit FROM __THIS__

On native these all work. On codegen you get UnsupportedInCodegen. So switching engines changes which SQL is legal — the same query that ran yesterday stops running when you flip the flag.

The important part: this FAILS LOUDLY. Codegen raises rather than silently computing something different. These are not bugs and never produced a wrong answer — they are honest "not implemented yet" refusals, and tests/test_codegen_coverage.py pins the exact set so nothing drifts in silently.

WHAT THIS TICKET DOES
Close the deferred surface so the opt-in engine stops being a downgrade. The gap was 16 skips in the differential suite (native + the DataFusion oracle cover and pass all of it), inventoried by QA on 2026-07-19:

  Container surface (~13 of 16) — struct/list column projection, struct field access, struct/list construction (named_struct / array), struct/list comparison, UNNEST
  Operators (2)                 — unary minus on a non-literal (-a), and the || string-concat operator

Each item landing shrinks the skip set and updates test_codegen_coverage.py, which currently pins the exact list.

PRIORITY CONTEXT
Low, and deliberately so. decision-7 ruled native default / codegen opt-in, which settled the precondition (AC#3) — this is a fast-follow for people who opt into codegen, not milestone work. If codegen were ever promoted toward default, this becomes real feature-completeness and the priority rises.

Ordering note from PM: the two operator defers are cheap scalar work and were never truly framing-gated (same category as CASE, which shipped on codegen regardless), so they went first; the container surface is the real body of the ticket.

Context: doc-9 (rich type system and UNNEST), doc-8 (composition — deferred slices).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 each deferred case passes on the codegen backend (the 16-skip set shrinks)
- [ ] #2 tests/test_codegen_coverage.py updated as items land (it currently pins the exact skip set)
- [x] #3 PRECONDITION: codegen-engine framing decision made (default vs opt-in) before this is actively prioritized
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-19 15:43
---
QA (Fermi) confirmed the exact 16-skip inventory on current master (2026-07-19): unnest() x5, struct field access x3, struct/list-typed column x3, struct/list construction x1, struct/list comparison x1, named_struct() x1, unary-minus-on-non-literal x1, || operator x1. Split insight to unblock the framing gate: the TWO operator-level defers (unary minus on a non-literal, ||) are cheap scalar ops that are NOT truly framing-gated — same category as CASE (TASK-30), which shipped on codegen regardless of the default-vs-opt-in question. The other 14 (struct/list/UNNEST container surface) are the real framing-gated core. Not spinning a separate ticket yet (no demand for the 2 ops today; native covers them), but documented here so the easy wins can be pulled ahead of the framing decision if codegen-default is chosen or demand appears.
---

author: Iris (PM)
created: 2026-07-19 16:13
---
decision-7 RULED (2026-07-19): native is default, codegen is opt-in for now. So AC #3's precondition ('framing decision made') is satisfied — decided → opt-in → this stays LOW, a fast-follow only for someone who opts into codegen, not milestone work. Reclassified from 'blocked on an open question' to 'deprioritized by decision' (dropped the 'blocked' label). Revisit only if codegen gets promoted toward default.
---

author: Iris (PM)
created: 2026-07-19 16:18
---
Dispatched to Ritchie (2026-07-19). AmirHossein's call: Ritchie takes the codegen lane (TASK-29 then TASK-34), Wren takes the rest (m-1 spine). Precondition AC #3 checked — decision-7 is ruled (opt-in), so this is cleared to work. Goal is codegen feature-completeness for opt-in users; still Low importance vs the native default path, but actively worked now. Suggested order within the ticket: the 2 cheap operator defers (unary-minus-on-non-literal, ||) first, then the container surface (struct/list/named_struct/UNNEST).
---
<!-- COMMENTS:END -->
