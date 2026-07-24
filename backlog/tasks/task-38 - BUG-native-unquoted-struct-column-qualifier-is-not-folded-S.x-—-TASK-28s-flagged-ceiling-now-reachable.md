---
id: TASK-38
title: >-
  BUG native: unquoted struct-column qualifier is not folded (S.x) — TASK-28's
  flagged ceiling, now reachable
status: To Do
assignee:
  - Wren
created_date: '2026-07-24 00:36'
updated_date: '2026-07-24 21:25'
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
Fails LOUDLY (raises "Unknown column: S") rather than silently computing a different value, so it is in the TASK-37 category rather than the TASK-36 category. Medium: real, but self-announcing, and the workaround (lowercase or quote the qualifier) is available once you know. Native is the DEFAULT serving engine, which is what keeps it from being Low.

RELATED: TASK-37 (native has no struct(...)/make_array(...) dispatch) and TASK-36 (native does not widen mixed-numeric list elements) are the other native container gaps from the same Phase B work. Possibly worth scoping all three as one "native container parity" push rather than three separate fixes — they are all in expr_build.rs / types.rs and all surfaced together.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Expr::Column carries the qualifier's quote information end-to-end (change table: Option<String> to a quote-preserving representation, or add a table_quoted flag) through expr_build -> plan, so quote_style is no longer discarded before the struct-column branch can use it
- [ ] #2 At the struct-column field-access fallback (plan.rs:1077), an UNQUOTED qualifier folds to lowercase (S -> s) while a QUOTED qualifier stays case-exact ("S" stays S) — matching DataFusion and consistent with TASK-28. The AC to get exactly right: fold unquoted, never fold quoted
- [ ] #3 Relation qualifiers (__THIS__/__STATE__/generated joins) UNAFFECTED BY THIS CHANGE — must NOT fold here, full suite stays green (the naive fold broke 96 tests). This AC is about not BREAKING the relation branch; its OWN separate oracle inversion is out of scope, tracked in DRAFT-22
- [ ] #4 The xfail_on_native marker on tests/test_diff_types.py::test_uppercase_qualifier_field_access is removed and the test passes on both engines vs the DataFusion oracle, in the same commit as the fix
- [ ] #5 TASK-28 AC#5's ceiling note in expr_build.rs is updated or removed — the gap it flagged 'unreachable' is now closed, so a stale unreachable comment would misdescribe reality
- [ ] #6 AC#6 (measure the sibling relation-alias case) DISCHARGED: measured during recon, found a separate relation-branch inversion, filed as DRAFT-22. Nothing further required in TASK-38
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-24 02:32
---
COVERAGE CONFIRMED (2026-07-24): this is Wren's GROUP C — test_uppercase_qualifier_field_access. Already captured; no new ticket needed. Provenance: came in with TASK-29 Phase B (d8e56e9, fb20afe, 8d398bf), NOT from PR #16.

Wren independently suggested linking this to TASK-28 so whoever takes it starts from that precedent rather than rediscovering it — already done: this ticket records that it is TASK-28's AC#5 ceiling ('a real CamelCase table/struct-column qualifier is NOT folded — unreachable today ... flagged for when qualified tables become reachable'), which Phase B's struct field access made reachable. AC#4 already requires updating that ceiling note in expr_build.rs. Two people arriving at the same linkage independently is a good sign the ticket is pointed the right way.

IMPLEMENTER NOTES (from Wren, 2026-07-24) — read before starting:
1. REQUIRES RUST CHANGES (src/expr_build.rs). `uv sync` does NOT recompile Rust — you need `uv run maturin develop` to rebuild _interpreter. The TASK-33 guard (953c726) auto-rebuilds when src/*.rs is newer than the .pyd, but only before tests.
2. Do NOT run `cargo test` in this environment — it fails with an unrelated pyo3 STATUS_DLL_NOT_FOUND. Not your bug; do not chase it.
3. The test is xfail(strict=True), so it FAILS LOUDLY the moment the gap closes. Flip the xfail off IN THE SAME COMMIT as the fix, or the suite goes red on success.
---

author: Iris (PM)
created: 2026-07-24 02:35
---
Promoted from draft and assigned to Wren (2026-07-24, AmirHossein's go). QUEUE POSITION 4 of 4. Shares src/expr_build.rs with TASK-37 — do them back-to-back. Note AC#4: this one also requires updating TASK-28's AC#5 ceiling note in expr_build.rs, since the gap it flagged is being closed rather than staying unreachable.
---

author: Iris (PM)
created: 2026-07-24 21:21
---
SCOPE DECISION — AmirHossein ruled OPTION (a): the full fix, not the one-line version the ticket implied (2026-07-24). This materially expands TASK-38 from 'fold the qualifier in expr_build.rs' to a core Expr::Column change carrying quote_style through to plan.rs. ACs rewritten to (a).

WHY the ticket's original framing was wrong (Wren, measured):
- The naive one-line fold (expr_build.rs:44, fold parts[0]) BREAKS 96 TESTS. Our own rewrite emits internally-qualified SQL (SELECT __THIS__.age / __STATE__.avg_age ... FROM __THIS__ LEFT JOIN __STATE__), so folding parts[0] turns __THIS__ -> __this__ and every windowed transform on native dies.
- expr_build CANNOT distinguish a relation qualifier from a struct-column qualifier — that is only known later in plan.rs::validate_expr (resolved.get(t)). Relation -> use as-is (don't fold). Not-a-relation -> struct-field fallback at plan.rs:1077 (should fold). But by then Expr::Column stores table as a bare Option<String> and quote_style is gone. Hence (a): carry the quote info so the struct-column branch can fold unquoted / preserve quoted.

This is why the ticket said 'unreachable' at TASK-28 time and why the honest fix is bigger. AmirHossein chose correctness over the stopgap. Added AC#6: measure whether a CamelCase TABLE alias hits the same quote-loss (Wren flagged it unmeasured) — fix in scope or follow-up, don't leave it hanging. Priority stays Medium (user-facing severity unchanged; scope grew, importance didn't).
---

author: Iris (PM)
created: 2026-07-24 21:25
---
AC#6 DISCHARGED by measurement (Wren, 2026-07-24, before writing code — exactly the instruction). It found MORE than the CamelCase-alias case: native's relation-qualifier handling is INVERTED vs the oracle on quoted forms (DataFusion registers the relation under a folded name, so quoted-exact \"__THIS__\" MISSES on the oracle while native accepts it; quoted-lower is the reverse). Full measurements in DRAFT-22.

SCOPE CALL (mine, Wren-agreed): the relation-qualifier inversion stays OUT of TASK-38 and is filed as DRAFT-22 (Low). Reasons: (1) it's the RELATION branch, TASK-38 is the STRUCT-COLUMN branch — different code, different fix; (2) fixing it means matching DataFusion's register-time relation folding, a second core change in plan.rs relation resolution on top of TASK-38's Expr::Column shape change; (3) one core change per PR. And it's LOW: measured unreachable through the public API — SQLTransform.fit raises loudly or agrees; only a direct internal InferFn call hits the inversion, and always loudly, never a silent wrong value.

AC#3 REWORDED for honesty (Wren's catch): the old wording implied the relation branch is currently CORRECT. It is not — it's inverted. AC#3 now says relation qualifiers are unaffected BY THIS CHANGE (don't break them, per the 96-test finding) and points to DRAFT-22 for their own divergence. Two different claims that the AC had conflated.

Wren is proceeding with TASK-38's brainstorm+plan for the struct-column branch only — unaffected by this call either way, so no reason to make him wait.
---
<!-- COMMENTS:END -->
