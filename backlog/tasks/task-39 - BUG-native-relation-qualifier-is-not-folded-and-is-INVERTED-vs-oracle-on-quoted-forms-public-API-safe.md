---
id: TASK-39
title: >-
  BUG native: relation qualifier is not folded and is INVERTED vs oracle on
  quoted forms (public-API-safe)
status: To Do
assignee: []
created_date: '2026-07-24 21:24'
updated_date: '2026-07-24 21:26'
labels:
  - native
  - parity
  - bug
  - identifier-folding
  - internal-api
dependencies:
  - TASK-38
references:
  - src/plan.rs
  - TASK-38
  - TASK-28
documentation:
  - doc-1 (DataFusion function catalogue — parity oracle)
priority: low
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by Wren while measuring TASK-38's AC#6 (2026-07-24), BEFORE writing code. Measurements are his, direct against both engines — not inferred.

THE DIVERGENCE
native's RELATION-qualifier handling is not merely unfolded — it is INVERTED vs the DataFusion oracle on quoted forms:

                      ORACLE                      NATIVE
  __this__.age        [{'v':7}]                   Unknown column: __this__     DIVERGE
  \"__this__\".age      [{'v':7}]                   Unknown column: __this__     DIVERGE
  \"__THIS__\".age      No field named \"__THIS__\"   [{'v':7}]                   DIVERGE (inverted)
  __THIS__.age        [{'v':7}]                   [{'v':7}]                   agree
  __THIS__.AGE        [{'v':7}]                   [{'v':7}]                   agree

ROOT CAUSE
DataFusion REGISTERS the relation under a FOLDED name: ctx.from_arrow(name=\"__THIS__\") is stored as __this__. So on the oracle, unquoted __THIS__ folds and hits, quoted-lower \"__this__\" hits, quoted-exact \"__THIS__\" MISSES. Native compares the qualifier raw, giving the exact opposite answer on both quoted forms. Same for a user table `t`: oracle accepts T.age and \"t\".age, rejects \"T\".age; native inverts.

WHY LOW — UNREACHABLE THROUGH THE PUBLIC API (measured via SQLTransform)
  SELECT __this__.age FROM __THIS__    -> fit() RAISES ValueError 'Column qualifier __this__ does not reference...' (clean, loud, at build time)
  SELECT \"__THIS__\".age FROM __THIS__  -> batch=[1,2,3] infer=[1,2,3], agree
Only a direct InferFn(...) call — an INTERNAL API, not the user surface — reaches the inverted behaviour, and it fails LOUDLY, never a silent wrong value. So user impact today is nil; this is correctness-of-the-engine-internals, not a user-facing parity gap.

WHY SEPARATE FROM TASK-38 (Wren's reasoning, PM-agreed)
1. Different branch: TASK-38 is the STRUCT-COLUMN branch (plan.rs:1077 fallback). This is the RELATION branch (plan.rs relation resolution). Different code, different fix.
2. The fix here is to match DataFusion's REGISTER-TIME folding of relation names — a second core change on top of TASK-38's Expr::Column quote-carrying, touching how plan.rs resolves relations.
3. Standing rule: one core change at a time. TASK-38 is already a core Expr::Column shape change; do not stack a plan.rs relation-resolution change on the same PR.

DEPENDS ON TASK-38: TASK-38 builds the Expr::Column quote-carrying this leans on, and AC#4 here requires TASK-38's struct-column folding to stay green. Do this AFTER TASK-38 lands.

WHEN PICKED UP: pin with an xfail-strict differential test first (the standing native-bug process), then fix. Not pinned yet — discovered by ad-hoc measurement during TASK-38 recon, deliberately not added to TASK-38's branch to keep concerns separate.

Priority Low: public API is safe and loud (see the measurements above). Revisit priority if a user-facing path ever starts reaching InferFn's raw qualifier handling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 native folds relation qualifiers to match DataFusion's register-time folding: unquoted T -> t hits, quoted-lower "t" hits, quoted-exact "T" misses (mirroring the oracle's stored folded name)
- [ ] #2 The quoted-form inversion is gone: "__THIS__".age and "__this__".age give the same answer on native as on the oracle
- [ ] #3 A differential test pins relation-qualifier folding across unquoted/quoted-lower/quoted-exact for both a generated relation (__THIS__) and a user table, passing on both engines
- [ ] #4 No regression to the struct-column-branch folding delivered in TASK-38, and the 96-test windowed-transform suite stays green
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-24 21:26
---
Promoted from draft to a tracked task (AmirHossein, 2026-07-24) — the relation-qualifier inversion is the natural follow-up to TASK-38's struct-column fix. Stays Low and UNASSIGNED for now; it is the sibling relation-branch change (register-time folding in plan.rs) and reads best as the next native-parity ticket after TASK-38 lands, since both touch the qualifier-resolution path and TASK-38 builds the quote-carrying that this can lean on. Not dispatched — awaiting AmirHossein's go on who/when.
---
<!-- COMMENTS:END -->
