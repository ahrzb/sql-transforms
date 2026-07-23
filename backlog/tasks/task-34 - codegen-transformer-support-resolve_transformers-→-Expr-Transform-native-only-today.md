---
id: TASK-34
title: >-
  codegen: transformer support (resolve_transformers → Expr::Transform,
  native-only today)
status: To Do
assignee:
  - Ritchie
created_date: '2026-07-19 16:08'
updated_date: '2026-07-23 04:41'
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
Codegen has NO transformer support — transformer refs are a native-only feature. SQLTransform.infer/infer_batch runs on the native InferFn, and the resolve_transformers rewrite (SQL call → Expr::Transform) lives only in the native path; codegen raises UnsupportedInCodegen for it (verified by TASK-31: the CASE-branch transformer test asserts codegen DEFERS loudly rather than computing). This is a whole feature class codegen lacks, distinct from TASK-29's deferred expr/container surface (which explicitly excludes transformers). Placeholder ticket so the gap is tracked and this 'do we have a ticket?' question stops recurring — NOT active work. BLOCKED-ON-FRAMING: only worth building if codegen becomes a maintained/default serving path (decision-7, the two-engine framing question, still open/parked). Bigger lift than TASK-29 — it's the transformer-resolution machinery, not a few expr shapes. Until decision-7 lands, native carries transformers and codegen defers them loudly (correct, not a bug).
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
<!-- COMMENTS:END -->
