---
id: TASK-34
title: >-
  codegen: transformer support (resolve_transformers → Expr::Transform,
  native-only today)
status: In Progress
assignee:
  - Ritchie
created_date: '2026-07-19 16:08'
updated_date: '2026-07-24 14:19'
labels:
  - codegen
  - transformer
dependencies:
  - TASK-29
references:
  - sql_transform/_codegen/
  - decision-7
  - TASK-29
documentation:
  - doc-7 (Transformer execution model)
  - 'doc-8 (Composition — {transform}(col) references)'
priority: low
type: feature
ordinal: 34000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
You wrote a transform that uses a fitted sklearn transformer — the library's headline feature:

    SELECT {scaler}(age, income) AS scaled FROM __THIS__

It works. Then you opt into the codegen engine for serving and it refuses: UnsupportedInCodegen. Not a subset of the syntax — the ENTIRE transformer-ref feature is unavailable. A user who builds their pipeline around transformer refs simply cannot use codegen at all.

Like TASK-29's gaps, this fails loudly rather than silently miscomputing (verified by TASK-31: the CASE-branch transformer test asserts codegen DEFERS rather than computing a wrong answer). Correct behavior, not a bug — but a whole feature class the opt-in engine lacks.

WHAT THIS TICKET DOES
Give codegen its own transformer resolution. Today the resolve_transformers rewrite (SQL call -> Expr::Transform) lives only in the native path — SQLTransform.infer/infer_batch runs on the native InferFn, and codegen has no equivalent. This would build that machinery on the codegen side, with transform == infer parity against the DataFusion/native oracle, and flip TASK-31's "defers loudly" assertion into a real parity case.

Distinct from TASK-29, which covers deferred EXPRESSION and CONTAINER surface and explicitly excludes transformers. This is the bigger lift of the two — transformer-resolution machinery, not a handful of expression shapes.

PRIORITY CONTEXT
Low. decision-7 ruled native default / codegen opt-in, which satisfied the precondition (AC#1) — so this is decided-low rather than blocked-on-an-open-question. It is a tracked placeholder so the recurring "do we have a ticket for this?" question stops. Only becomes real work if codegen is ever promoted toward being a maintained/default serving path.

Depends on TASK-29. Pre-authorized by AmirHossein to dispatch to Ritchie automatically once TASK-29 lands.

Context: doc-7 (transformer execution model), doc-8 (composition — {transform}(col) references).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PRECONDITION: codegen-engine framing decision made (decision-7, default vs opt-in) before this is actively prioritized — do not build while codegen is the parked/experimental engine
- [ ] #2 codegen resolves transformer refs to its own Expr::Transform equivalent and computes them (not UnsupportedInCodegen), with transform == infer parity vs the DataFusion/native oracle
- [ ] #3 the TASK-31 codegen 'defers loudly' assertion is flipped to a real parity case once codegen computes transformers
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-19 16:14
---
decision-7 RULED (2026-07-19): native default, codegen opt-in. AC #1's precondition is now satisfied (decided → opt-in), so this is decided-LOW rather than blocked-on-open. Stays a tracked placeholder assigned to Ritchie; not active work unless codegen is promoted to default. Dropped the 'blocked' label to match.
---

author: Iris (PM)
created: 2026-07-19 16:18
---
Queued for Ritchie AFTER TASK-29 (dependency). Precondition AC #1 checked — decision-7 ruled opt-in, so cleared to work once TASK-29 lands. Not dispatched yet; Ritchie is on TASK-29 first.
---

author: Iris (PM)
created: 2026-07-23 04:41
---
PRE-AUTHORIZED (2026-07-23): AmirHossein green-lit dispatching TASK-34 to Ritchie automatically once TASK-29 lands. No further approval needed — PM dispatches on TASK-29 completion. (Dependency TASK-29 still In Progress.)
---

author: Iris (PM)
created: 2026-07-24 14:19
---
Auto-dispatched to Ritchie (2026-07-24) on TASK-29 closing, per AmirHossein's standing pre-authorization — no further approval needed. TASK-29 dependency satisfied (PR #17 merged, master a83b742). Ritchie starting with a spec/plan check under docs/superpowers/ before code, which is correct. Precondition AC#1 (decision-7) was already satisfied.
---
<!-- COMMENTS:END -->
