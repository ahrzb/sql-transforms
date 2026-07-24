---
id: TASK-35
title: >-
  Build the transformer-callout struct in feature_names_in_ order — make the
  order-desync bug class unrepresentable
status: In Progress
assignee:
  - Wren
created_date: '2026-07-24 02:32'
updated_date: '2026-07-24 14:27'
labels:
  - transformer-refs
  - parity
  - design-fix
milestone: m-1
dependencies: []
references:
  - sql_transform/_transformer_ref.py
  - sql_transform/_transformer_udf.py
  - 'src/expr.rs:414'
  - 'tests/test_diff_transformer_callout.py:70'
  - 'PR #16'
documentation:
  - 'doc-8 (Composition — {transform}(col) references)'
  - doc-7 (Transformer execution model)
priority: high
type: enhancement
ordinal: 35000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ORIGIN: AmirHossein's insight (2026-07-24), routed via Wren. Measured by Wren before filing, not theorised.

THE PROBLEM: WE SCRAMBLE, THEN UNSCRAMBLE TWICE, IN TWO LANGUAGES

    1. _transformer_ref.py::_named_struct   BUILDS the struct from call.expressions
                                            -- i.e. the USER's SQL call order
    2. _transformer_udf.py                  reorders back to feature_names_in_
                                            cols = [struct_array.field(f) for f in feature_names]
    3. src/expr.rs:414                      reorders back AGAIN, in Rust
                                            for feat in input_features { fields.iter().find(...) }

We control step 1. So we choose an order, then spend two independent implementations — one Python, one Rust — repairing the mismatch we ourselves created. The question that motivates this ticket: why be sensitive to an order we pick?

WHY THIS IS WORTH DOING (the evidence)
THREE of the four blocking bugs found across TASK-3's review rounds were THE SAME DEFECT: a call-order vs fitted-order desync between the struct we build and the schema we declare. Twice, the fix for one instance introduced the next.

That is the signature of a design problem, not a coding problem. With one order used everywhere, the class stops being fixed-but-fragile and becomes UNREPRESENTABLE — you cannot disagree with yourself when there is only one order to get wrong. This ticket buys the elimination of a bug class that has already cost four review rounds.

MEASURED
Wren ran the change against the full suite before this was filed:

    _order = {c.name: c for c in call.expressions}
    call.set("expressions", [_named_struct([_order[f] for f in feat])])
    in_schema, out_schema, y = _probe(obj, feat, table)   # in_schema matches the struct

    -> 561 passed, 5 skipped, 5 xfailed  (IDENTICAL to the current baseline)

The experiment was restored; PR #16 does not contain it. So this is a known-viable change, not a hopeful one.

=== CRITICAL SCOPE CAVEAT — READ BEFORE IMPLEMENTING ===
DO NOT DELETE THE ENGINE-SIDE REORDERS (steps 2 and 3).

They become no-ops for the {sc}(...) authoring path, but they MUST STAY. Raw SQL can hand-author a struct in any order — tests/test_diff_transformer_callout.py:70 does exactly that:

    "SELECT __tfm_0__(named_struct('income', income, 'age', age)) ..."

so the engine-side reorder remains the real defense for hand-authored SQL. THE WIN IS THAT THE AUTHORING PATH STOPS GENERATING MISMATCHES FOR IT TO REPAIR — not that the repair goes away. An implementer who reads this ticket as "remove the duplicate reorders" would break hand-authored SQL.

ORTHOGONAL — DO NOT CONFLATE
This does NOT change user-visible semantics. {sc}(income, age) still equals {sc}(age, income) afterwards, because the reorder happens at build time.

Whether it SHOULD stay equal is a separate, deliberately-unresolved question: sklearn REFUSES a reordered DataFrame ("Feature names must be in the same order as they were in fit") while we silently reorder. AmirHossein has seen that question and set it aside. It is NOT part of this ticket and must not be folded in.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 _named_struct builds the struct in feature_names_in_ order rather than the user's SQL call order
- [ ] #2 in_schema is derived from that same order, so the struct and the declared schema cannot disagree
- [ ] #3 Engine-side reorders in _transformer_udf.py and src/expr.rs:414 are RETAINED, each with a comment stating why (hand-authored SQL can supply any order)
- [ ] #4 A test proves a hand-authored out-of-order named_struct still works (the tests/test_diff_transformer_callout.py:70 shape), so the retained defense is covered rather than merely present
- [ ] #5 transform == infer parity green on both engines (DataFusion oracle, decision-1)
- [ ] #6 A test pins that {sc}(income, age) and {sc}(age, income) still agree — build-time reorder means no user-visible semantic change
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-24 02:35
---
Dispatched to Wren (2026-07-24), AmirHossein's explicit go, and explicitly FIRST — ahead of the three native container tasks (TASK-36/37/38), which are queued behind it. Rationale for that ordering: TASK-35 eliminates a bug class that has already cost four review rounds, and it touches the transformer-ref authoring path Wren has full context on from TASK-2/TASK-3. The native tasks are Rust work in a different area and can wait.
---

author: Iris (PM)
created: 2026-07-24 14:25
---
Delivered as PR #18 (OPEN, not merged) — NOT closing until merged; ACs verified against the merged diff then. Scope verified now against the PR diff: 2 files (_transformer_ref.py + test), ZERO Rust — the scope caveat held, engine-side reorders untouched. Wren reports 574 passed / 12 xfailed.

AC-VERIFICATION FINDING (why I'm not pre-ticking AC#3): AC#3 has two parts — reorders RETAINED (done: _transformer_udf.py and src/expr.rs unchanged) AND 'each with a comment stating why'. The existing comments at both sites explain WHAT the reorder does ('Reorder the struct's fields to feature_names_in_ order'), not the post-TASK-35 rationale the AC intends: that the {sc}(...) authoring path no longer needs it but it STAYS because hand-authored named_struct can supply any order. Without that 'do not delete — hand-authored SQL relies on this' note, a future reader could reasonably see the reorder as dead code TASK-35 obsoleted.

Proportionate view: the REAL guard against deletion is the test (AC#4, test_struct_field_order_independence_parity) — removing a reorder turns it red. So this is belt-and-suspenders documentation, not a correctness gap. Flagged to Wren to add a one-line 'why this stays' comment at both sites while the branch is open, since it is literally the AC wording and nearly free. Not a merge blocker on its own; AmirHossein's approval stands regardless.
---

author: Iris (PM)
created: 2026-07-24 14:27
---
AC#3 comment-clause CLOSED on PR #18 (commit 9e71469). Both reorder sites (_transformer_udf.py + src/expr.rs) now carry the 'DO NOT DELETE — hand-authored SQL relies on it; the {t}(...) path emits fitted order post-TASK-35 so this is a no-op for it but the only defense for hand-authored named_struct' rationale, cross-referencing each other. Comment-only, 574 passed / 12 xfailed unchanged. So AC#3 is met to the letter, not 90%. All six ACs now satisfied on the branch; still verifying against the MERGED diff before I tick them and close — PR #18 awaits AmirHossein's approval. (The src/expr.rs comment tripped the expected maturin rebuild, clean.)
---
<!-- COMMENTS:END -->
