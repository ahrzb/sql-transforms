---
id: doc-5
title: 'SQL-conformance harness (design brief, parked / out of scope)'
type: other
created_date: '2026-07-18 23:28'
---
**Status: OUT OF SCOPE — parked as design, not a milestone (AmirHossein 2026-07-19).**
Was a milestone (archived) with steps S1–S4 (archived: former TASK-9/12/10/11). The
rationale is ratified and stays live as [[decision-6]]; this doc preserves the step-level
build plan so the milestone can be re-opened cheaply when there's room. Not lost, just not
in the current work queue.

## What it is

Run DataFusion's own `.slt` sqllogictest query corpus through **both serving engines**
(native `InferFn` + codegen) vs the **live DataFusion oracle**, asserting *values* match.
Imports DataFusion's edge coverage (e.g. `substr` has ~30 upstream boundary/unicode/overrun
cases where we hand-wrote ~3) instead of hand-authoring it. Observation-only, zero engine
changes; value divergences it surfaces become tickets (same process as native/codegen parity
bugs). See [[decision-6]] for the full rationale and the load-bearing choices:
oracle-truth-not-golden-text, allowlist-driven extraction, the "steal-the-spec" DoD hook, and
"a menu, not a to-do" (we do NOT chase slt pass-rate — VISION rejects becoming a SQL engine).

## Build plan (S1–S4, when re-scoped)

- **S1 — parser + manifest + `--refresh-slt`.** Test-only package. Parse `.slt`, extract
  `query` records only (no DDL/statement). Manifest maps `our_function -> (source file,
  upstream revision, case hashes)`. `--refresh-slt` re-fetches upstream at a pinned ref and
  rebumps. Seed corpus = `expr.slt` @ DataFusion `67947b6` (48.0.0-rc2).
- **S2 — allowlist extractor + dual-engine runner.** Extractor pulls query records whose
  function set ⊆ the supported allowlist and are self-contained (no FROM/DDL/aggregate) →
  runnable-by-construction, near-zero skips. Runner executes each through native + codegen +
  live DataFusion, asserting values match (oracle-truth; [[decision-1]] + [[decision-2]]).
  Seed allowlist = current shipped surface:
  upper/lower/trim/concat/coalesce/nullif/abs/round/substr/substring/struct + operators +
  cast to numeric/string/bool.
- **S3 — nullability-tolerant comparator.** DataFusion constant-folds literal exprs to
  not-null; our engines emit nullable. Comparator asserts values match while tolerating
  null-flag differences (only values must match — [[decision-2]]). Genuine value divergence
  still fails.
- **S4 — wire the "steal-the-spec" DoD hook.** Make "add function to allowlist + re-extract"
  a repeatable, low-friction step so future feature work auto-extends conformance coverage.
  Documented + scripted path: implement fn → add to allowlist → re-extract → its DataFusion
  cases land in the suite.

## Relationships

- Precursor pattern for the multi-language runtimes' generative differential corpus — see [[doc-4]].
- De-risks the native-per-transformer swaps by widening the parity net to the full
  scalar surface.
