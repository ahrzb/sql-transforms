---
id: TASK-36
title: >-
  BUG native: mixed int/float list elements are not widened (silent value
  divergence)
status: To Do
assignee:
  - Wren
created_date: '2026-07-23 14:30'
updated_date: '2026-07-24 02:35'
labels:
  - native
  - parity
  - bug
  - containers
dependencies: []
references:
  - src/types.rs
  - tests/test_diff_types.py
documentation:
  - doc-9 (Rich type system and UNNEST — status and deferred edges)
  - doc-1 (DataFusion function catalogue — parity oracle)
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
You bundle two features into one list column — a count and a ratio:

    SELECT [bedrooms, price_per_sqft] AS dims FROM __THIS__

`bedrooms` is an int column, `price_per_sqft` is a float column.

At training time you call transform() over the DataFrame and get:   [3.0, 12.5]
At serving time you call infer() on a single row and get:           [3, 12.5]

No error. No warning. The int element silently stays an int on the serving path and becomes a float on the training path. Anything downstream that is dtype-sensitive — a numpy cast, a model expecting float64, a serialized feature vector compared against a stored schema — now sees different data in training than in production. This is the failure mode the whole differential harness exists to prevent: you only find it when the model misbehaves in prod, and nothing in the stack points at the list literal.

Realistic triggers: any list feature mixing an integer column with a float column — [count, rate], [year_built, lot_ratio], [n_visits, avg_spend]. Users won't think of these as "mixed-type" — they're just two numbers.

ROOT CAUSE (measured, not inferred)
native's unify_list_element_types (src/types.rs) is EXACT-EQUALITY-ONLY, so it refuses to widen mixed numeric elements to a common type. DataFusion (the oracle) and codegen both widen to list<double>.

    SELECT [x, y] AS l FROM t   -- x int = 1, y float = 2.5
      DataFusion (oracle) + codegen:  [1.0, 2.5]   (widened to list<double>)
      native:                         [1, 2.5]     (un-widened)

WHY THIS IS THE SERIOUS ONE
Unlike the struct/make_array dispatch gap (TASK-37), this is NOT a missing capability that errors out — native accepts the query and returns a DIFFERENT VALUE. Silent cross-engine divergence on the DEFAULT serving engine (decision-7). Proposed High on that basis.

SAME BUG CLASS as the one Ritchie just fixed on the codegen side in infer_type's ListExpr arm (unifying element bases via _common_base, like COALESCE). The native fix mirrors it: compute a common numeric base instead of demanding exact equality.

Surfaced by Ritchie's TASK-29 container work (2026-07-23), pinned by a strict xfail_on_native in tests/test_diff_types.py::test_list_construct_mixed_numeric_widens. Filed per the standing native-bug process — Ritchie correctly flagged rather than fixing inline. DataFusion is the oracle (decision-1).

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 native unify_list_element_types (src/types.rs) computes a common numeric base for mixed elements instead of exact equality, mirroring codegen's _common_base/COALESCE logic
- [ ] #2 SELECT [x, y] with int + float elements yields [1.0, 2.5] on native, matching the DataFusion oracle
- [ ] #3 The xfail_on_native marker on tests/test_diff_types.py::test_list_construct_mixed_numeric_widens is removed and the test passes on both engines
- [ ] #4 Root-cause sweep: confirm no other native type-unification site is exact-equality-only where DataFusion widens (the same class of silent divergence elsewhere)
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-24 02:32
---
COVERAGE CONFIRMED (2026-07-24): this is Wren's GROUP B — test_list_construct_mixed_numeric_widens. Already captured; no new ticket needed. The xfail's own reason string says 'separate parity bug, own ticket', which is exactly this. Provenance: came in with TASK-29 Phase B (d8e56e9, fb20afe, 8d398bf), NOT from PR #16.

Wren's framing agrees with the split I made: this is a SEMANTICS decision about numeric widening, whereas TASK-37 is plain missing dispatch — different files, different root causes, correctly separate tickets.

IMPLEMENTER NOTES (from Wren, 2026-07-24) — read before starting:
1. REQUIRES RUST CHANGES (src/types.rs). `uv sync` does NOT recompile Rust — you need `uv run maturin develop` to rebuild _interpreter. The TASK-33 guard (953c726) auto-rebuilds when src/*.rs is newer than the .pyd, but only before tests.
2. Do NOT run `cargo test` in this environment — it fails with an unrelated pyo3 STATUS_DLL_NOT_FOUND. Not your bug; do not chase it.
3. The test is xfail(strict=True), so it FAILS LOUDLY the moment the gap closes. Flip the xfail off IN THE SAME COMMIT as the fix, or the suite goes red on success.
---

author: Iris (PM)
created: 2026-07-24 02:35
---
Promoted from draft and assigned to Wren (2026-07-24, AmirHossein's go). QUEUE POSITION 2 of 4 — after TASK-35, before TASK-37/38. First of the three native container tasks because it is the only SILENT-wrong-value one (the other two fail loudly).
---
<!-- COMMENTS:END -->
