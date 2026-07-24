---
id: DRAFT-12
title: 'native: no dispatch for struct(...) and make_array(...) construction'
status: Draft
assignee: []
created_date: '2026-07-23 14:30'
updated_date: '2026-07-24 02:32'
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

SEVERITY vs DRAFT-13
This is a missing capability that fails loudly, not a wrong answer — bad, but self-announcing. DRAFT-13 (mixed-numeric list widening) is the silent-wrong-value one and is rated higher.

WHY IT MATTERS: native is the DEFAULT serving engine (decision-7), so a surface that works on the opt-in engine but not the default is backwards. Medium rather than High because the bracket literal gives users a working alternative for lists, and struct construction has no demonstrated demand yet.

Surfaced by Ritchie's TASK-29 container work (2026-07-23), pinned by 3 strict xfail_on_native markers in tests/test_diff_types.py. Filed per the standing native-bug process (xfail-strict + ticket, never fix inline).

DRAFT pending AmirHossein's review of scope/priority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 native convert_function (src/expr_build.rs) dispatches struct(...) positional form, naming fields c0/c1/... exactly as DataFusion does
- [ ] #2 native dispatches struct(a AS x, ...) named form (exp.PropertyEQ) with explicit field names
- [ ] #3 native dispatches make_array(...) to the same construction path the bracket literal [a, b] already uses
- [ ] #4 The 3 xfail_on_native markers in tests/test_diff_types.py (test_struct_construct_positional, test_struct_construct_named, test_make_array_construct) are removed and the tests pass on both engines against the DataFusion oracle
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
<!-- COMMENTS:END -->
