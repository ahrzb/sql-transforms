---
id: decision-6
title: slt conformance corpus and DoD hook
date: '2026-07-18 15:39'
status: accepted
---
## Context

Our per-function edge coverage is thin — `substr` has ~3 assertions where DataFusion
enumerates ~30 boundary/unicode/overrun cases. DataFusion authors those cases in its
own sqllogictest (`.slt`) files. Hand-authoring equivalent coverage is wasteful when
the oracle already ships it. Blessed by AmirHossein 2026-07-18 (Tester proposal, Wren
scoped).

## Decision

**Adopt DataFusion's `.slt` files as a revision-pinned *query corpus* (not golden
text).** Run each extracted query through **both serving engines** (native `InferFn` +
codegen) **and the live DataFusion oracle**, asserting *values* match.

- **Oracle-truth, not golden text.** We do NOT diff against the slt `----` expected
  block (that couples us to DataFusion's output *formatting* — NULL/float/bool
  rendering — which rots). slt files are used purely as a corpus of queries. This is
  exactly [[decision-1]] (DataFusion is the oracle) + [[decision-2]] (only values
  match, not formatting/null-flags).
- **Allowlist-driven extraction.** Pull only self-contained `query` records whose
  function/construct set ⊆ the supported allowlist — runnable-by-construction,
  near-zero skips. The allowlist is the single knob.
- **"Steal-the-spec" DoD hook.** New scalar-function work **must** extend the allowlist
  + re-extract, so that function's DataFusion conformance cases land automatically.
  Coverage grows on the same cadence as features.
- **A menu, not a to-do.** The gap-analysis feature list is a demand-driven reservoir;
  we implement a function when a transform needs it. We do **not** chase slt pass-rate
  as a driver — that would pull us toward reimplementing DataFusion's catalog, which
  VISION explicitly rejects (we're a transform lib, not a SQL engine).

## Consequences

- Near-term infra milestone (SQL-conformance harness — **now parked, see doc-5**),
  parallel to the native-swap phase. Steps S1–S4 (now archived: former TASK-9/12/10/11).
- Harness is **observation-only, zero engine changes**; value divergences it surfaces
  become tickets (same process as TASK-1 native bugs / TASK-4 codegen bugs).
- Feature tasks that add a scalar function gain the allowlist-extend DoD item.
- Directly de-risks the native-swap phase's native-per-transformer swaps by widening the parity net to
  the full scalar surface.
