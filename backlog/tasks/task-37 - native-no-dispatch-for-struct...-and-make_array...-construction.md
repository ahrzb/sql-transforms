---
id: TASK-37
title: 'native: no dispatch for struct(...) and make_array(...) construction'
status: To Do
assignee:
  - Wren
created_date: '2026-07-23 14:30'
updated_date: '2026-07-24 14:36'
labels:
  - native
  - parity
  - sql-surface
  - containers
dependencies: []
references:
  - src/expr_build.rs
  - tests/test_diff_types.py
documentation:
  - doc-9 (Rich type system and UNNEST — status and deferred edges)
  - doc-1 (DataFusion function catalogue — parity oracle)
priority: medium
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
You copy a snippet from the DataFusion docs (or write what feels natural) to pack fields together:

    SELECT struct(bedrooms AS beds, baths AS baths) AS layout FROM __THIS__
    -- or
    SELECT make_array(lat, lon) AS coords FROM __THIS__

You call fit() and transform() on your training frame. Both work — that path runs on DataFusion.
Then you deploy and call infer() for single-row serving. It fails: native doesn't recognize the function at all.

So the feature works in your notebook and breaks in production. The user did nothing wrong — they used valid DataFusion SQL that the library accepted at fit time.

The confusing part: for LISTS there is a spelling that works and one that doesn't, with nothing to tell them apart —

    SELECT [lat, lon] AS coords    -- works on both engines today
    SELECT make_array(lat, lon)    -- works on transform(), fails on infer()

For STRUCTS there is no working alternative spelling at all on native.

ROOT CAUSE
native's convert_function (src/expr_build.rs) has no case for these container-construction forms, so it doesn't recognize them as functions. Codegen supports them and matches the DataFusion oracle; native does not:

    SELECT struct(a, b) AS s        -> DataFusion/codegen: {c0: 1, c1: 2} (positional field names). Native: no dispatch.
    SELECT struct(a AS x, b AS y)   -> DataFusion/codegen: {x: 1, y: 2} (parses as exp.PropertyEQ). Native: no dispatch.
    SELECT make_array(a, b) AS l    -> DataFusion/codegen: [1, 2]. Native: no dispatch.

NOT affected: the bracket literal `[a, b]` DOES reach native's list path and passes today (tests/test_diff_types.py::test_list_construct). The gap is specifically the FUNCTION-call spellings, not list/struct construction as a concept.

SEVERITY vs TASK-36
This is a missing capability that fails loudly, not a wrong answer — bad, but self-announcing. TASK-36 (mixed-numeric list widening) is the silent-wrong-value one and is rated higher.

WHY IT MATTERS: native is the DEFAULT serving engine (decision-7), so a surface that works on the opt-in engine but not the default is backwards. Medium rather than High because the bracket literal gives users a working alternative for lists, and struct construction has no demonstrated demand yet.

Surfaced by Ritchie's TASK-29 container work (2026-07-23), pinned by 3 strict xfail_on_native markers in tests/test_diff_types.py. Filed per the standing native-bug process (xfail-strict + ticket, never fix inline).

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 STRUCT construction handled via a SqlExpr::Struct arm in convert_expr (positional -> c0/c1/...; named `a AS x` -> field x), mirroring codegen's _convert_struct at plan.py:390 — NOT via convert_function
- [ ] #2 The now-identified DEAD `struct` branch in convert_function (expr_build.rs:197-204) is removed, since sqlparser routes struct(...) to SqlExpr::Struct and it never reaches convert_function
- [ ] #3 make_array(...) dispatched in convert_function -> Expr::List, the same construction path the bracket literal [a, b] already uses (mirrors codegen's exp.Array -> ListExpr at plan.py:395). make_array is the pinned case; include the array(...) alias ONLY if it is the same one-line dispatch AND gets its own parity assertion — otherwise leave it out, do not add untested surface
- [ ] #4 The 3 xfail_on_native markers (test_struct_construct_positional, test_struct_construct_named, test_make_array_construct) are removed and pass on both engines vs the DataFusion oracle; each flip in the same commit as its fix
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-24 02:32
---
COVERAGE CONFIRMED (2026-07-24): this is Wren's GROUP A — all 3 struct/make_array dispatch xfails (test_struct_construct_positional, test_struct_construct_named, test_make_array_construct). Already fully captured; no new ticket needed. Provenance verified: these came in with TASK-29 Phase B (d8e56e9, fb20afe, 8d398bf), NOT from PR #16.

IMPLEMENTER NOTES (from Wren, 2026-07-24) — read before starting:
1. REQUIRES RUST CHANGES (src/expr_build.rs). `uv sync` does NOT recompile Rust — you need `uv run maturin develop` to rebuild _interpreter. The TASK-33 guard (953c726) auto-rebuilds when src/*.rs is newer than the .pyd, but only before tests.
2. Do NOT run `cargo test` in this environment — it fails with an unrelated pyo3 STATUS_DLL_NOT_FOUND. Not your bug; do not chase it.
3. The tests are xfail(strict=True), so they FAIL LOUDLY the moment the gap closes. Flip each xfail off IN THE SAME COMMIT as the fix, or the suite goes red on success. That is the intended design, not a surprise.
---

author: Iris (PM)
created: 2026-07-24 02:35
---
Promoted from draft and assigned to Wren (2026-07-24, AmirHossein's go). QUEUE POSITION 3 of 4. Shares src/expr_build.rs with TASK-38 — worth doing back-to-back while the file is in context, though they stay separate tickets with separate xfail flips.
---

author: Iris (PM)
created: 2026-07-24 14:36
---
ROOT-CAUSE CORRECTION + AC UPDATE (2026-07-24), from Wren's measured recon before implementing. The ticket's original 'native has no struct(...) construction dispatch (expr_build.rs)' was imprecise — there are TWO distinct gaps, not one:

1. STRUCT is NOT a dispatch gap. sqlparser parses struct(...) into a first-class SqlExpr::Struct AST node, so it never reaches convert_function; it hits convert_expr's catch-all 'Unsupported expression'. convert_function ALREADY has a `struct` case (expr_build.rs:197-204) but it is DEAD CODE — unreachable. Real fix: a SqlExpr::Struct arm in convert_expr (mirrors codegen's _convert_struct), plus removing the dead branch. ACs #1/#2 rewritten to this.

2. MAKE_ARRAY genuinely IS a function-dispatch gap as the ticket said — reaches convert_function, hits the catch-all Ok(Expr::Function{}), fails at eval 'Unknown function'. Fix: a case mapping make_array -> Expr::List. Cross-parser note: sqlparser gives make_array as a Function, sqlglot gives the bracket form as exp.Array, so native's trigger is the function name where codegen's is the AST node.

MY RULINGS on Wren's three questions (within-ticket scope calls, PM's to make):
(a) REMOVE the dead struct branch in convert_function — yes. Dead code identified is dead code deleted; folded into AC#2.
(b) array(...) alias: make_array is the pinned/tested case and the AC. Include array() ONLY if it is the identical one-line dispatch and you add a parity assertion for it; otherwise out of scope, not a new ticket unless it turns out non-trivial. Don't add untested surface for completeness' sake.
(c) all three are error-out (loud) gaps, lower-stakes than TASK-36's silent one — acknowledged, no change; xfail-strict flip still applies per-test.

This is the validate-don't-assume pattern catching an imprecise ticket before code, same as Ritchie's spec corrections. The ticket is better for it.
---
<!-- COMMENTS:END -->
