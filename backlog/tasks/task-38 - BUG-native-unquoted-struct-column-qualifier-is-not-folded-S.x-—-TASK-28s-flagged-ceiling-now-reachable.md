---
id: TASK-38
title: >-
  BUG native: unquoted struct-column qualifier is not folded (S.x) — TASK-28's
  flagged ceiling, now reachable
status: To Do
assignee:
  - Wren
created_date: '2026-07-24 00:36'
updated_date: '2026-07-24 02:35'
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
- [ ] #1 native folds an unquoted struct-column qualifier to lowercase, matching DataFusion (SELECT S.x resolves against struct column s)
- [ ] #2 A quoted qualifier stays case-exact, consistent with the TASK-28 folding rule for ordinary identifiers
- [ ] #3 The xfail_on_native marker on tests/test_diff_types.py::test_uppercase_qualifier_field_access is removed and the test passes on both engines
- [ ] #4 TASK-28 AC#5's ceiling note in expr_build.rs is updated or removed, since the gap it flagged is now closed rather than merely unreachable
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
<!-- COMMENTS:END -->
