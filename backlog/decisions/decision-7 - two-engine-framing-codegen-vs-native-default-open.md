---
id: decision-7
title: 'Two-engine framing: codegen vs native as default — OPEN'
date: '2026-07-19'
status: proposed
---
## Context

The native `InferFn` interpreter is the current serving engine and the parity baseline
(oracle = DataFusion, [[decision-1]]). A **codegen engine** was subsequently built
(`worktree-codegen-inferfn`, `sql_transform/_codegen/`, plan
`superpowers/plans/2026-07-17-codegen-inferfn.md`, 11 tasks): suite 397 passed / 14
skipped (containers) / 3 xfailed (pinned native divergences), parity target = the
DataFusion oracle. The old "codegen / compiled inference path" was parked as *likely
won't do, revisit only with a benchmark in hand* — that condition is now satisfied (the
n=1 boundary-bound benchmark) and an engine actually exists.

## Decision

**OPEN — not yet ratified.** Whether codegen is adopted as a maintained / default path
vs. the native interpreter is AmirHossein's pending framing call. This record captures
the artifact + the open question, not a decision.

This framing is load-bearing downstream:
- Blocks TASK-29 (codegen deferred SQL surface) and the framing context on TASK-4.
- The multi-language inference runtimes epic ([[doc-4]]) rests on it — that design treats
  the native Rust engine as **one-of-N**, which only holds if two-engine is the accepted
  shape.

## Consequences / notes

- Codegen's adversarial review surfaced **2 codegen-only parity divergences** (native
  already matched the oracle — no native ticket): float→string for |x| < 1e-4
  (`CAST(1e-5 AS VARCHAR)` → DF `'0.00001'`, codegen `'1e-05'`); integer arithmetic
  overflow (`9223372036854775807 * 2` → DF/native wrap to `-2`, codegen Python-bigint
  `18446744073709551614`). **Both fixed & merged into codegen (`131fa0b`); TASK-4 Done.**
- If oracle parity on the codegen path is wanted, codegen also needs the shared native
  edge-case fixes (the historical `SUBSTR`/`NaN`/`CAST` set); it already fixed `ROUND(int)`
  and `COALESCE(int,float)` typing.
- Until this is ratified, codegen stays an artifact on its branch, not a default.
